"""
app1.py - Flask Web Application for Adaptive-MMRAG Framework
"""

# ============================================================
# OFFLINE LOCK — Must be the very first code executed.
# Forces HuggingFace / SentenceTransformers / Transformers to
# use ONLY locally cached model files.  If these vars are set
# after any ML library is imported it is already too late.
# ============================================================
import os
os.environ['HF_HUB_OFFLINE'] = '1'          # HuggingFace Hub: never ping the internet
os.environ['TRANSFORMERS_OFFLINE'] = '1'    # Transformers: local cache only
os.environ['HF_DATASETS_OFFLINE'] = '1'     # datasets library: local only
os.environ.setdefault('PYTHONUTF8', '1')    # Force UTF-8 output on Windows

import nlp_summarizer
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import shutil
import csv
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template_string, stream_with_context, send_from_directory, send_file
try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama_module = None
    OLLAMA_AVAILABLE = False

# Provide a safe ollama shim so code using ollama.chat / ollama.generate
# does not NameError if the package is absent.
class _OllamaUnavailable:
    """Stub that raises a clear ConnectionError for every Ollama call."""
    def __getattr__(self, name):
        def _stub(*args, **kwargs):
            raise ConnectionError(
                "Ollama Python package is not installed in this Python environment. "
                "Run: python -m pip install ollama"
            )
        return _stub

ollama = _ollama_module if OLLAMA_AVAILABLE else _OllamaUnavailable()

from werkzeug.utils import secure_filename
import uuid


# Import pipeline components
from multimodal_unified_parallel import ParallelMasterPipeline
from ragchatbot import (
    CrossModalRAGChatbot,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
)
from adaptive_controller import AdaptiveRAGController
from adaptive_config import ADAPTIVE_MODE_DEFAULT
from tts_handler import tts_engine

# Flask app configuration
app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except Exception:
    pass
app.secret_key = 'change-this-secret-key-in-production'

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_output'
app.config['METRICS_FOLDER'] = 'metrics'
app.config['CLIP_MODEL_PATH'] = 'SIH/clip-model'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'doc',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
    'mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'wma', 'aac'
}

# Global state
chatbot_instance = None
adaptive_controller = AdaptiveRAGController()
processing_lock = False
current_session_id = None

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['METRICS_FOLDER'], exist_ok=True)


def _startup_warmup():
    """
    Bug 1 Fix: Pre-load Ollama model into RAM at server startup (no upload needed).
    Runs in background daemon thread so Flask starts instantly.
    Ensures the model is warm before the first user query arrives.
    """
    try:
        import time as _time
        _time.sleep(2)  # Give Flask a moment to fully start
        print("⚡ [Startup Warmup] Pre-loading Ollama model into RAM...")
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt="Hello",
            options={"num_predict": 1, "temperature": 0},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        print("⚡ [Startup Warmup] ✅ Ollama model is warm and ready. First query will be fast.")
    except Exception as e:
        print(f"⚡ [Startup Warmup] Notice (non-critical): {e}")


# Fire startup warmup immediately (no upload needed)
threading.Thread(target=_startup_warmup, daemon=True).start()



def _warmup_models(chatbot):
    """
    Background warmup: pre-load Ollama model and FAISS reranker into memory
    so the first user query is as fast as subsequent ones.
    Runs in a daemon thread immediately after upload completes.
    """
    try:
        print("⚡ [Warmup] Starting background model warmup...")
        # 1. Warm FAISS + CrossEncoder reranker with a trivial search
        chatbot.search_by_text("hello", k=1, use_reranker=False)
        print("⚡ [Warmup] FAISS index warmed.")
        # 2. Warm Ollama — generate exactly 1 token so the model loads into RAM
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt="Hi",
            options={"num_predict": 1, "temperature": 0},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        print("⚡ [Warmup] Ollama model loaded and ready. First query will be fast.")
    except Exception as e:
        print(f"⚡ [Warmup] Notice (non-critical): {e}")


def append_metrics_row(name: str, row: dict):
    """Persist metrics in both CSV and JSONL for later model comparison."""
    csv_path = os.path.join(app.config['METRICS_FOLDER'], f'{name}.csv')
    jsonl_path = os.path.join(app.config['METRICS_FOLDER'], f'{name}.jsonl')

    if name == 'query_metrics':
        total = row.get('total_query_duration_seconds')
        if total is None:
            total = row.get('total_latency_seconds', 0.0)
        row['total_query_duration_seconds'] = total
        row['total_latency_seconds'] = total

        gen = row.get('generation_duration_seconds')
        if gen is None:
            gen = row.get('generation_latency_seconds', 0.0)
        row['generation_duration_seconds'] = gen
        row['generation_latency_seconds'] = gen

        ret = row.get('retrieval_duration_seconds')
        if ret is None:
            ret = row.get('retrieval_latency_seconds', 0.0)
        row['retrieval_duration_seconds'] = ret
        row['retrieval_latency_seconds'] = ret

    csv_row = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            csv_row[k] = json.dumps(v, ensure_ascii=False)
        else:
            csv_row[k] = v

    csv_exists = os.path.exists(csv_path)
    existing_headers = None
    if csv_exists:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
        except Exception:
            existing_headers = None

    if not csv_exists or not existing_headers:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_row.keys()))
            writer.writeheader()
            writer.writerow(csv_row)
    else:
        new_keys = [k for k in csv_row.keys() if k not in existing_headers]
        if new_keys:
            all_headers = existing_headers + new_keys
            with open(csv_path, 'r', encoding='utf-8') as f:
                old_rows = list(csv.DictReader(f))
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_headers)
                writer.writeheader()
                for r in old_rows:
                    writer.writerow(r)
                writer.writerow(csv_row)
        else:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=existing_headers, extrasaction='ignore')
                writer.writerow(csv_row)

    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')


def get_top_sources(search_results, limit: int = 5) -> str:
    """Capture the most relevant distinct sources for evaluation logs."""
    sources = []
    for result in search_results[:limit]:
        source = result['metadata'].get('source', 'Unknown')
        if source not in sources:
            sources.append(source)
    return '|'.join(sources)


def print_panel_results_table():
    """Print a panel-friendly benchmark table in a distinct terminal style."""
    lines = [
        "",
        "=" * 92,
        " RESULTS SUMMARY",
        "=" * 92,
        "MMORE benchmark ",
        "-" * 92,
        f"{'Method':<18} {'Blue BLEU':>10} {'Blue ROUGE-L':>14} {'Blue CER':>10} "
        f"{'Gatsby BLEU':>12} {'Gatsby ROUGE-L':>16} {'Gatsby CER':>12}",
        "-" * 92,
        f"{'MMORE':<18} {'0.8608':>10} {'0.9940':>14} {'0.0241':>10} "
        f"{'0.7973':>12} {'0.9813':>16} {'0.0295':>12}",
        f"{'MMORE (fast)':<18} {'0.8639':>10} {'0.9963':>14} {'0.0206':>10} "
        f"{'0.0000':>12} {'0.0000':>16} {'1.0000':>12}",
        f"{'Docling':<18} {'0.8643':>10} {'0.9959':>14} {'0.0199':>10} "
        f"{'0.5451':>12} {'0.6582':>16} {'0.5518':>12}",
        "-" * 92,
        "Reference books: The Blue Castle (digital PDF) | The Great Gatsby (scanned images)",
        "=" * 92,
        ""
    ]
    print("\n".join(lines))


def load_html():
    """Load the HTML file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Error: index.html not found</h1>
        <p>Please ensure index.html is in the same directory as app.py</p>
        </body>
        </html>
        """


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_all_data():
    """Clear all uploaded and processed files."""
    global chatbot_instance, current_session_id
    
    # Clear uploads safely
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
        except Exception:
            pass
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Clear processed safely
    if os.path.exists(app.config['PROCESSED_FOLDER']):
        try:
            shutil.rmtree(app.config['PROCESSED_FOLDER'], ignore_errors=True)
        except Exception:
            pass
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    chatbot_instance = None
    nlp_summarizer.clear_profiles()
    current_session_id = None


@app.route('/')
def index():
    """Serve the main HTML page."""
    html_content = load_html()
    return render_template_string(html_content)


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and processing."""
    global chatbot_instance, processing_lock, current_session_id
    
    try:
        request_started_at = time.perf_counter()
        print("\n" + "="*70)
        print("“¤ UPLOAD REQUEST RECEIVED")
        print("="*70)
        
        # Check if already processing
        if processing_lock:
            print("  System busy")
            return jsonify({
                'success': False,
                'error': 'System is busy processing files. Please wait.'
            }), 429
        
        # Check files
        if 'files' not in request.files:
            print(" No files in request")
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        print(f"“¦ Received {len(files)} files")
        
        if not files or files[0].filename == '':
            print(" No files selected")
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Start processing
        processing_lock = True
        
        # Clear previous data
        print("§¹ Clearing previous data...")
        clear_all_data()
        
        # Create new session
        current_session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], current_session_id)
        os.makedirs(session_dir, exist_ok=True)
        print(f"“ Session: {current_session_id}")
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                print(f"    Saved: {filename}")
        
        if not uploaded_files:
            processing_lock = False
            print(" No valid files")
            return jsonify({
                'success': False,
                'error': 'No valid files uploaded'
            }), 400
        
        print(f"\nš€ Processing {len(uploaded_files)} files...")
        
        # Process files
        pipeline = ParallelMasterPipeline(
            clip_model_path=app.config['CLIP_MODEL_PATH'],
            output_dir=app.config['PROCESSED_FOLDER'],
            max_workers=None
        )
        
        results = pipeline.run_complete_pipeline(session_dir)
        
        # Initialize chatbot with FULL capabilities (reusing cached singletons)
        print("\n¤– Initializing enhanced chatbot...")
        chatbot_instance = CrossModalRAGChatbot(
            index_path=results['index_path']
        )

        # Bug 6 Fix: Run warmup SYNCHRONOUSLY before returning so the first query is instant.
        # This adds ~1-2s to the upload response time but guarantees model is hot when UI shows "Ready".
        print("⚡ [Post-Upload Warmup] Ensuring model is warm before returning...")
        _warmup_models(chatbot_instance)

        # NLP Automatic Document Summarization & Keyphrase Extraction (Option 2)
        print("\n🔍 Running NLP document summarization & keyphrase extraction...")
        nlp_profiles = {}
        try:
            nlp_profiles = nlp_summarizer.process_corpus(chatbot_instance.metadata)
            print(f"✅ NLP Profiling complete for {len(nlp_profiles)} documents")
        except Exception as nlp_err:
            print(f"⚠️  NLP profiling notice: {nlp_err}")

        append_metrics_row('upload_metrics', {
            'timestamp': datetime.now().isoformat(),
            'session_id': current_session_id,
            'model_name': OLLAMA_MODEL,
            'uploaded_file_count': len(uploaded_files),
            'uploaded_files': '|'.join(uploaded_files),
            'total_chunks': results['total_chunks'],
            'pipeline_duration_seconds': round(results['duration_seconds'], 3),
            'total_request_duration_seconds': round(time.perf_counter() - request_started_at, 3),
            'workers': results['workers'],
            'chunks_per_second': round(results['chunks_per_second'], 3),
            'index_path': results['index_path']
        })
        
        processing_lock = False
        
        print("\n UPLOAD & PROCESSING COMPLETE")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': f'Processed {results["total_chunks"]} chunks successfully',
            'files': uploaded_files,
            'total_chunks': results['total_chunks'],
            'duration': f"{results['duration_seconds']:.1f}s",
            'nlp_profiles': nlp_profiles
        }), 200
    
    except Exception as e:
        processing_lock = False
        print(f"\n UPLOAD FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/documents/summary', methods=['GET'])
def get_document_summaries():
    """Return cached NLP document profiles (keyphrase tags + summaries)."""
    try:
        profiles = nlp_summarizer.get_all_summaries()
        return jsonify({
            'success': True,
            'summaries': profiles
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """
    Handle text queries with ENHANCED accuracy.
    
    KEY IMPROVEMENTS:
    1. Uses hybrid search with reranking (k=15 for better coverage)
    2. Leverages full context formatting from ragchatbot.py
    3. Proper error handling and logging
    4. Uses the comprehensive prompt that works in CLI
    5. Returns debug info for transparency
    """
    global chatbot_instance
    
    try:
        print("\n" + "="*70)
        print("’¬ QUERY REQUEST (ENHANCED)")
        print("="*70)
        
        # Get JSON data
        data = request.get_json(silent=True)
        if not data:
            print(" No JSON data")
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        query_text = data.get('query', '').strip()
        debug_mode = data.get('debug', False)  # Optional debug flag
        
        if not query_text:
            print(" Empty query")
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400

        request_started_at = time.perf_counter()

        if chatbot_instance is None:
            # Fallback: general chat mode using Ollama (no RAG context)
            print("?????? Chatbot not initialized. Using general chat mode.")
            generation_started_at = time.perf_counter()
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI assistant. Provide a complete, accurate, and structured answer. "
                            "If the user asks for a list or specific count of items (such as 10 items, commands, or steps), "
                            "provide all of them completely with clear, numbered bullet points and concise explanations. "
                            "Never stop halfway or omit requested items."
                        ),
                    },
                    {"role": "user", "content": query_text},
                ],
                options=OLLAMA_OPTIONS,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            generation_duration = time.perf_counter() - generation_started_at
            answer = response["message"]["content"]

            append_metrics_row('query_metrics', {
                'timestamp': datetime.now().isoformat(),
                'session_id': current_session_id or 'no_session',
                'mode': 'general_chat',
                'model_name': OLLAMA_MODEL,
                'query': query_text,
                'retrieval_duration_seconds': 0.0,
                'generation_duration_seconds': round(generation_duration, 3),
                'total_query_duration_seconds': round(time.perf_counter() - request_started_at, 3),
                'retrieved_results': 0,
                'top_sources': '',
                'answer_length_chars': len(answer)
            })

            return jsonify({
                'success': True,
                'answer': answer
            }), 200

        print(f"???? Query: {query_text}")
        print(f"???? Debug mode: {debug_mode}")
        print("???? Performing hybrid search with reranking...")

        retrieval_started_at = time.perf_counter()
        # Six reranked chunks provide enough PDF context while keeping the
        # prompt small enough for responsive local generation.
        search_results = chatbot_instance.search_by_text(query_text, k=6)
        retrieval_duration = time.perf_counter() - retrieval_started_at

        if not search_results:
            append_metrics_row('query_metrics', {
                'timestamp': datetime.now().isoformat(),
                'session_id': current_session_id or 'no_session',
                'mode': 'rag',
                'model_name': OLLAMA_MODEL,
                'query': query_text,
                'retrieval_duration_seconds': round(retrieval_duration, 3),
                'generation_duration_seconds': 0.0,
                'total_query_duration_seconds': round(time.perf_counter() - request_started_at, 3),
                'retrieved_results': 0,
                'top_sources': '',
                'answer_length_chars': 0
            })
            return jsonify({
                'success': True,
                'answer': 'I could not find relevant information in the indexed files for that question.'
            }), 200

        context = chatbot_instance.format_context_by_modality(search_results)
        print(
            f"???? Retrieved {len(search_results)} results | "
            f"text={len(context['text'])}, tables={len(context['tables'])}, "
            f"images={len(context['images'])}, audio={len(context['audio'])}"
        )

        generation_started_at = time.perf_counter()
        answer = chatbot_instance.generate_answer(query_text, search_results, query_type="text")
        generation_duration = time.perf_counter() - generation_started_at

        append_metrics_row('query_metrics', {
            'timestamp': datetime.now().isoformat(),
            'session_id': current_session_id or 'no_session',
            'mode': 'rag',
            'model_name': OLLAMA_MODEL,
            'query': query_text,
            'retrieval_duration_seconds': round(retrieval_duration, 3),
            'generation_duration_seconds': round(generation_duration, 3),
            'total_query_duration_seconds': round(time.perf_counter() - request_started_at, 3),
            'retrieved_results': len(search_results),
            'top_sources': get_top_sources(search_results),
            'answer_length_chars': len(answer)
        })

        # Get search results for debug info
        if debug_mode:
            debug_info = {
                'num_results': len(search_results),
                'context_breakdown': {
                    'text_chunks': len(context['text']),
                    'tables': len(context['tables']),
                    'images': len(context['images']),
                    'image_files': len(context['image_paths']),
                    'audio': len(context['audio'])
                },
                'top_sources': list(set([
                    r['metadata'].get('source', 'Unknown') 
                    for r in search_results[:5]
                ])),
                'top_scores': [
                    f"{r['score']:.4f}" for r in search_results[:5]
                ]
            }
            print(f"???? Debug info: {debug_info}")

        response = {
            'success': True,
            'answer': answer
        }

        print_panel_results_table()
        
        if debug_mode:
            response['debug'] = debug_info
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n QUERY FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error message
        error_msg = str(e)
        if "requires more system memory" in error_msg.lower():
            error_msg = (
                f"The Ollama model '{OLLAMA_MODEL}' does not fit in available memory. "
                "Close memory-heavy applications or set OLLAMA_MODEL to a smaller installed model."
            )
        elif "ollama" in error_msg.lower():
            error_msg = " Ollama is not running. Please start it with: ollama serve"
        elif "connection" in error_msg.lower():
            error_msg = " Cannot connect to Ollama. Ensure it's running on localhost:11434"
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/query/stream', methods=['POST'])
def query_stream():
    """Stream response tokens (supports both general chat and RAG modes)."""
    global chatbot_instance, current_session_id
    
    data = request.get_json(silent=True) or {}
    query_text = data.get('query', '').strip()
    if not query_text:
        return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400

    request_started_at = time.perf_counter()
    # Bug 10: Read conversation history sent from frontend (last 6 turns)
    conversation_history = data.get('history', [])

    if chatbot_instance is None:
        # No documents indexed — respond in general chat mode with a banner.
        # If Ollama is not running, give a clear actionable error instead of a silent failure.
        @stream_with_context
        def generate_no_documents_response():
            if not OLLAMA_AVAILABLE:
                yield (
                    "> ❌ **[Ollama Not Installed]**\n\n"
                    "The Ollama Python package is not installed in this Python environment.\n"
                    "Fix: `python -m pip install ollama`, then restart the server."
                )
                return

            try:
                generation_started_at = time.perf_counter()
                full_answer = []
                yield "> ℹ️ **[General Chat — No documents uploaded yet. Upload files to enable RAG.]**\n\n"

                # Bug 4 Fix: Stronger anti-hallucination system prompt for general chat
                # Bug 10 Fix: Include conversation history for multi-turn memory
                system_msg = (
                    'You are a helpful, honest AI assistant with broad general knowledge. '
                    'Answer questions accurately and helpfully. '
                    'IMPORTANT RULES:\n'
                    '1. If the user\'s question contains typos or grammar errors, silently correct and answer the intended question.\n'
                    '2. For factual questions about specific local places, people, or recent events you are not confident about, '
                    'say "I\'m not fully certain, but..." rather than inventing details.\n'
                    '3. Never confuse cities, regions, or countries. If unsure about a specific local fact, say so.\n'
                    '4. Provide clear, structured, accurate answers based on what you reliably know.'
                )
                # Build messages with history for multi-turn memory
                messages = [{'role': 'system', 'content': system_msg}]
                for turn in (conversation_history or [])[-6:]:
                    role = turn.get('role', 'user')
                    content = str(turn.get('content', '')).strip()
                    if content and role in ('user', 'assistant'):
                        messages.append({'role': role, 'content': content[:800]})
                messages.append({'role': 'user', 'content': query_text})

                chunks = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                for chunk in chunks:
                    token = chunk['message']['content']
                    full_answer.append(token)
                    yield token

                generation_duration = time.perf_counter() - generation_started_at
                answer_text = "".join(full_answer)
                try:
                    append_metrics_row('query_metrics', {
                        'timestamp': datetime.now().isoformat(),
                        'session_id': current_session_id or 'no_session',
                        'mode': 'general_chat',
                        'model_name': OLLAMA_MODEL,
                        'query': query_text,
                        'retrieval_duration_seconds': 0.0,
                        'generation_duration_seconds': round(generation_duration, 3),
                        'total_query_duration_seconds': round(time.perf_counter() - request_started_at, 3),
                        'retrieved_results': 0,
                        'top_sources': '',
                        'answer_length_chars': len(answer_text)
                    })
                except Exception:
                    pass  # metrics failure must not crash the response

            except Exception as exc:
                exc_str = str(exc).lower()
                is_connection_issue = (
                    'connection' in exc_str
                    or 'winerror 10061' in exc_str
                    or 'refused' in exc_str
                    or 'ollama' in exc_str
                    or '11434' in exc_str
                    or 'failed to connect' in exc_str
                    or isinstance(exc, ConnectionError)
                )
                if is_connection_issue:
                    yield (
                        f"> ❌ **[Ollama Not Running]**\n\n"
                        f"Cannot reach Ollama at localhost:11434.\n\n"
                        f"**Fix:** Open a new terminal and run:\n```\nollama serve\n```\n"
                        f"Then make sure the model is pulled:\n```\nollama pull qwen2.5:3b\n```\n\n"
                        f"_(Connection error: {exc})_"
                    )
                else:
                    yield f"\n\n❌ Unexpected error during generation: {exc}"

        return Response(generate_no_documents_response(), mimetype='text/plain; charset=utf-8')



    else:
        # Adaptive-MMRAG streaming mode (with Baseline toggle support)
        try:
            req_mode = data.get('mode', 'adaptive' if ADAPTIVE_MODE_DEFAULT else 'baseline')
            req_response_language = data.get('response_language', 'en') or 'en'
            
            @stream_with_context
            def generate_adaptive_stream():
                try:
                    final_metadata = None
                    stream_gen = adaptive_controller.process_query_stream(
                        query=query_text,
                        chatbot_instance=chatbot_instance,
                        mode=req_mode,
                        response_language=req_response_language,
                        conversation_history=conversation_history  # Bug 10: multi-turn memory
                    )
                    
                    for token, metadata in stream_gen:
                        if metadata is not None:
                            final_metadata = metadata
                        if token:
                            yield token
                    
                    if final_metadata:
                        final_metadata['timestamp'] = datetime.now().isoformat()
                        final_metadata['session_id'] = current_session_id or 'no_session'
                        final_metadata['model_name'] = OLLAMA_MODEL
                        final_metadata['query'] = query_text
                        try:
                            append_metrics_row('query_metrics', final_metadata)
                        except Exception:
                            pass
                        print_panel_results_table()
                except Exception as exc:
                    exc_str = str(exc).lower()
                    is_connection_issue = (
                        'connection' in exc_str
                        or 'winerror 10061' in exc_str
                        or 'refused' in exc_str
                        or 'ollama' in exc_str
                        or '11434' in exc_str
                        or 'failed to connect' in exc_str
                        or isinstance(exc, ConnectionError)
                    )
                    if is_connection_issue:
                        yield (
                            f"\n\n> ❌ **[Ollama Not Running]**\n\n"
                            f"Cannot reach Ollama at localhost:11434.\n\n"
                            f"**Fix:** Open a new terminal and run:\n```\nollama serve\n```\n"
                            f"Then make sure the model is pulled:\n```\nollama pull qwen2.5:3b\n```\n\n"
                            f"_(Connection error: {exc})_"
                        )
                    else:
                        import traceback
                        tb = traceback.format_exc()
                        print(f"  [adaptive stream error] {exc}\n{tb}")
                        yield f"\n\n❌ Error during RAG generation: {exc}"

        
            return Response(generate_adaptive_stream(), mimetype='text/plain; charset=utf-8')

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/query/image-stream', methods=['POST'])
def query_image_stream():
    """
    Stream a response for an image-as-prompt query.
    Accepts multipart/form-data with:
      image: the attached image file
      query: optional text question (default: 'What is this about?')
      mode:  'adaptive' | 'baseline'
    Performs CLIP visual similarity search against the indexed database,
    then streams an LLM answer grounded in the matching records.
    """
    global chatbot_instance

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    img_file = request.files['image']
    if not img_file or img_file.filename == '':
        return jsonify({'success': False, 'error': 'Empty image file'}), 400

    query_text = request.form.get('query', '').strip() or 'What is this image about?'
    temp_img_path = None

    def _describe_image_as_text(img_path: str, orig_filename: str) -> str:
        """
        Image description using CLIP zero-shot classification (primary) + PIL metadata.
        CLIP actually identifies the visual content of the image by comparing it to
        a vocabulary of common object labels. Falls back to filename stem if CLIP fails.
        """
        from PIL import Image as PILImage

        # --- Primary: CLIP zero-shot classification ---
        clip_result = ""
        try:
            from vectorstore import classify_image_with_clip
            clip_result = classify_image_with_clip(img_path, top_k=3)
        except Exception:
            pass  # Fall through to filename approach

        # --- PIL metadata ---
        stem = os.path.splitext(os.path.basename(orig_filename))[0].replace('_', ' ').replace('-', ' ').title()
        try:
            with PILImage.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                img_format = img.format or 'JPEG'
            size_kb = os.path.getsize(img_path) // 1024
            meta = f"Dimensions: {w}x{h}px, {mode} {img_format}, {size_kb}KB"
        except Exception:
            meta = "Image file"

        # --- Build final description ---
        if clip_result:
            # CLIP succeeded — use visual classification as primary description
            return (
                f"CLIP Visual Classification: {clip_result}\n"
                f"Image Filename: {orig_filename} (filename subject: {stem})\n"
                f"Technical Info: {meta}\n"
                f"CONCLUSION: The image visually contains or depicts: "
                f"{clip_result.split(';')[0].split('(')[0].strip()}."
            )
        else:
            # Fallback: filename-based description
            return (
                f"Image Filename: {orig_filename}\n"
                f"Subject (from filename): {stem}\n"
                f"Technical Info: {meta}\n"
                f"CONCLUSION: Based on the filename, this image is about: {stem}."
            )

    try:
        # Save uploaded query image temporarily
        orig_filename = secure_filename(img_file.filename) or 'image.jpg'
        ext = os.path.splitext(orig_filename)[1] or '.jpg'
        temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"qimg_{uuid.uuid4().hex[:8]}{ext}")
        img_file.save(temp_img_path)

        # Build image description text — CLIP classification runs here
        img_description = _describe_image_as_text(temp_img_path, orig_filename)

        # Extract the visual subject for a direct, punchy prompt
        # e.g. "apple (high confidence)" → "apple"
        visual_subject = "the object in the image"
        try:
            if "CLIP Visual Classification:" in img_description:
                clip_line = img_description.split("CLIP Visual Classification:")[1].split("\n")[0].strip()
                visual_subject = clip_line.split("(")[0].strip().split(";")[0].strip()
            elif "Subject (from filename):" in img_description:
                visual_subject = img_description.split("Subject (from filename):")[1].split("\n")[0].strip()
        except Exception:
            pass

        if chatbot_instance is None:
            # No documents indexed — describe with CLIP + LLM
            @stream_with_context
            def stream_no_docs():
                yield "> ℹ️ **[Analyzing image content...]**\n\n"
                try:
                    img_prompt = (
                        f"The uploaded image shows: **{visual_subject}**.\n\n"
                        f"Full image analysis:\n{img_description}\n\n"
                        f"User question: {query_text}\n\n"
                        f"Answer directly and clearly. In your FIRST sentence, state that the image shows '{visual_subject}'. "
                        f"Do NOT say you cannot see the image — the image has been analyzed and the content is provided above.\n\n"
                        f"Answer:"
                    )
                    stream = ollama.generate(
                        model=OLLAMA_MODEL,
                        prompt=img_prompt,
                        stream=True,
                        options=OLLAMA_OPTIONS,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    for chunk in stream:
                        token = chunk.get('response', '')
                        if token:
                            yield token
                except Exception as e:
                    yield f"\n❌ Error analyzing image: {e}"
                finally:
                    if temp_img_path and os.path.exists(temp_img_path):
                        try: os.remove(temp_img_path)
                        except: pass
            return Response(stream_no_docs(), mimetype='text/plain; charset=utf-8')

        # Documents are indexed — do visual similarity search
        search_results = chatbot_instance.search_by_image(temp_img_path, k=8)

        if not search_results:
            # No similar content in DB — describe with CLIP + LLM
            @stream_with_context
            def stream_image_direct():
                yield "> ℹ️ **[Analyzing image content...]**\n\n"
                try:
                    img_prompt = (
                        f"The uploaded image shows: **{visual_subject}**.\n\n"
                        f"Full image analysis:\n{img_description}\n\n"
                        f"User question: {query_text}\n\n"
                        f"Answer directly. Your FIRST sentence must clearly state what the image depicts: '{visual_subject}'. "
                        f"Do NOT say you cannot analyze the image — analysis has already been done above.\n\n"
                        f"Answer:"
                    )
                    stream = ollama.generate(
                        model=OLLAMA_MODEL,
                        prompt=img_prompt,
                        stream=True,
                        options=OLLAMA_OPTIONS,
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                    for chunk in stream:
                        token = chunk.get('response', '')
                        if token:
                            yield token
                except Exception as e:
                    yield f"\n❌ Error analyzing image: {e}"
                finally:
                    if temp_img_path and os.path.exists(temp_img_path):
                        try: os.remove(temp_img_path)
                        except: pass
            return Response(stream_image_direct(), mimetype='text/plain; charset=utf-8')

        # Build grounded prompt from search results, enriched with CLIP image description
        context = chatbot_instance.format_context_by_modality(search_results)
        full_query = f"Based on the uploaded image (which shows: {visual_subject}), {query_text}"
        base_prompt = chatbot_instance.build_fast_rag_prompt(full_query, context)
        prompt = (
            f"[UPLOADED IMAGE — CLIP ANALYSIS]\n"
            f"This image shows: {visual_subject}\n"
            f"Full analysis: {img_description}\n\n"
            + base_prompt
        )

        @stream_with_context
        def stream_image_rag():
            try:
                response_stream = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                for chunk in response_stream:
                    token = chunk.get('response', '')
                    if token:
                        yield token
            except Exception as e:
                yield f"\n❌ Error generating image-based response: {e}"
            finally:
                if temp_img_path and os.path.exists(temp_img_path):
                    try: os.remove(temp_img_path)
                    except: pass

        return Response(stream_image_rag(), mimetype='text/plain; charset=utf-8')

    except Exception as e:
        if temp_img_path and os.path.exists(temp_img_path):
            try: os.remove(temp_img_path)
            except: pass
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/voice/transcribe', methods=['POST'])
def voice_transcribe():
    """Receive recorded voice audio blob, transcribe with offline Whisper.
    Returns both the original-language transcription and English translation.
    Form fields:
      audio: WAV blob
      mode:  'transcribe' (keep original) | 'translate' (to English, default)
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if not audio_file or audio_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty audio data'}), 400

        # Save to temp audio file
        ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        temp_voice_path = os.path.join(app.config['UPLOAD_FOLDER'], f"voice_{uuid.uuid4().hex[:8]}{ext}")
        audio_file.save(temp_voice_path)

        from audio_handler import VoiceQueryTranscriber
        transcriber = VoiceQueryTranscriber.get_instance()

        result = transcriber.transcribe_multilingual(temp_voice_path, task='transcribe')
        text = result.get('text', '').strip()

        # Cleanup temp file
        if os.path.exists(temp_voice_path):
            try:
                os.remove(temp_voice_path)
            except Exception:
                pass

        if text:
            print(f"🎙️ Voice Input Recognized: '{text}'")
            return jsonify({
                'success': True,
                'text': text
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Could not detect clear speech. Please speak again clearly.')
            }), 400

    except Exception as exc:
        print(f"❌ /voice/transcribe exception: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/voice/transcribe/partial', methods=['POST'])
def voice_transcribe_partial():
    """Quick partial transcription for real-time feedback during recording.
    Accepts a WAV blob and returns a fast transcription of whatever is audible.
    Used by the UI to show progressive text while the user is still speaking.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'text': ''}), 400
        
        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'success': False, 'text': ''}), 400

        ext = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"partial_{uuid.uuid4().hex[:6]}{ext}")
        audio_file.save(temp_path)

        from audio_handler import VoiceQueryTranscriber
        transcriber = VoiceQueryTranscriber.get_instance()
        result = transcriber.transcribe_multilingual(temp_path, task='transcribe')

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return jsonify({
            'success': result.get('success', False),
            'text': result.get('text', ''),
            'language': result.get('language', 'en'),
        }), 200

    except Exception as exc:
        return jsonify({'success': False, 'text': '', 'error': str(exc)}), 200  # 200 so UI doesn't break on partial failure


@app.route('/clear', methods=['POST'])
def clear():
    """Clear all data."""
    global chatbot_instance, processing_lock
    
    try:
        print("\n" + "="*70)
        print("§¹ CLEAR REQUEST")
        print("="*70)
        
        if processing_lock:
            print("  Cannot clear while processing")
            return jsonify({
                'success': False,
                'error': 'Cannot clear while processing'
            }), 429
        
        clear_all_data()
        
        print(" All data cleared")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': 'All data cleared successfully'
        }), 200
    
    except Exception as e:
        print(f"\n CLEAR FAILED: {e}\n")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Get system status with detailed info."""
    try:
        # Check Ollama server availability
        ollama_server_running = False
        ollama_models = []
        try:
            if OLLAMA_AVAILABLE:
                result = ollama.list()
                ollama_models = [m.model for m in (result.models or [])]
                ollama_server_running = True
        except Exception:
            pass

        status_info = {
            'ready': chatbot_instance is not None,
            'processing': processing_lock,
            'session_id': current_session_id,
            'ollama_package_installed': OLLAMA_AVAILABLE,
            'ollama_server_running': ollama_server_running,
            'ollama_models': ollama_models,
            'general_chat_available': OLLAMA_AVAILABLE and ollama_server_running
        }
        
        # Add index statistics if chatbot is ready
        if chatbot_instance is not None:
            from collections import Counter
            metadata = chatbot_instance.metadata
            
            types = Counter(m['type'] for m in metadata)
            modalities = Counter(m.get('modality', 'text') for m in metadata)
            
            status_info['statistics'] = {
                'total_chunks': len(metadata),
                'by_type': dict(types),
                'by_modality': dict(modalities)
            }
        
        return jsonify(status_info), 200
    
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500


# ============================================================================
# OFFLINE AUDIO GENERATION (TEXT-TO-SPEECH) & VOICE TRANSCRIPTION
# ============================================================================

@app.route('/tts/synthesize', methods=['POST'])
def tts_synthesize():
    """
    Synthesize assistant answer text into offline audio (WAV) using pyttsx3.
    Returns audio URL for streaming playback and download.
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    voice = data.get('voice', 'default')
    rate = data.get('rate')

    if not text:
        return jsonify({'success': False, 'error': 'No text provided for audio generation'}), 400

    try:
        success, result, spoken_text = tts_engine.synthesize(text, voice_preference=voice, rate=rate)
        if not success:
            return jsonify({'success': False, 'error': result}), 500

        audio_url = f"/tts/audio/{result}"
        return jsonify({
            'success': True,
            'audio_url': audio_url,
            'filename': result,
            'spoken_text': spoken_text,
            'voice': voice
        })
    except Exception as e:
        print(f"[TTS][ROUTE_ERROR] {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/tts/audio/<filename>', methods=['GET'])
def tts_serve_audio(filename):
    """Serve synthesized offline speech audio file."""
    try:
        # Sanitize filename
        safe_name = os.path.basename(filename)
        audio_dir = os.path.abspath(str(tts_engine.cache_dir))
        file_path = os.path.join(audio_dir, safe_name)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Audio file not found'}), 404
        return send_from_directory(audio_dir, safe_name, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    print(f" 404 Error: {request.path}")
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(e):
    print(f" 500 Error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large (max 500MB)'
    }), 413


if __name__ == '__main__':
    print("\n" + "="*70)
    print(">> ENHANCED MULTIMODAL RAG CHATBOT SERVER")
    print("\n KEY IMPROVEMENTS:")
    print("   Uses same search method as CLI (hybrid + reranking)")
    print("   Comprehensive prompt engineering")
    print("   Better context formatting by modality")
    print("   Enhanced error handling")
    print("   Detailed logging for debugging")
    print("\nFeatures:")
    print("   PDF, DOCX, Images, Audio support")
    print("   Parallel processing")
    print("   Cross-modal retrieval")
    print("   Natural language answers with citations")
    print("\nEndpoints:")
    print("  GET  /          - Main UI")
    print("  POST /upload    - Upload & process files")
    print("  POST /query     - Ask questions (ENHANCED)")
    print("  POST /clear     - Clear all data")
    print("  GET  /status    - System status")
    print("\nAccess at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Check if index.html exists
    if not os.path.exists('index.html'):
        print("  WARNING: index.html not found in current directory!")
        print("   Please ensure index.html is in the same folder as app.py\n")
    
    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
        print(" Ollama is running")
    except:
        print("  WARNING: Ollama might not be running!")
        print("   Start it with: ollama serve")
        print("   Pull LLaVA with: ollama pull llava:7b\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)


