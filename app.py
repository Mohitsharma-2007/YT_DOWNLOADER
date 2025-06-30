from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify, session, flash
from pytube import YouTube, Playlist, Search
from pytube.exceptions import VideoUnavailable, PytubeError, RegexMatchError
import os
import zipfile
from io import BytesIO
import threading
import time
import re
import logging
import subprocess
import json
import tempfile
import shutil
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import yt-dlp with error handling
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    logging.warning("yt-dlp not available. Will use pytube only.")
    
# Import OpenAI for AI summarization
try:
    import openai
    OPENAI_AVAILABLE = True
    # Set OpenAI API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
except (ImportError, Exception) as e:
    OPENAI_AVAILABLE = False
    logging.warning(f"OpenAI not available. AI summarization will be disabled. Error: {str(e)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AI Summarization function
def generate_video_summary(video_info):
    """
    Generate an AI summary of the video based on its title, description, and other metadata.
    
    Args:
        video_info: Dictionary containing video information
        
    Returns:
        A dictionary with summary information or error message
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OpenAI API key not set. Please add it to your .env file."
        }
    
    try:
        # Prepare the prompt with video information
        title = video_info.get('title', 'Unknown Title')
        description = video_info.get('description', 'No description available')
        author = video_info.get('author', 'Unknown')
        
        prompt = f"""
        Please provide a concise summary of this YouTube video based on its metadata:
        
        Title: {title}
        Channel: {author}
        Description: {description}
        
        Please include:
        1. Main topic or purpose of the video
        2. Key points or highlights
        3. Target audience
        4. Potential value or takeaways
        
        Format the summary in a clear, readable way with bullet points where appropriate.
        """
        
        # Get the model from environment or use default
        model = os.getenv("AI_MODEL", "gpt-3.5-turbo")
        
        # Call OpenAI API using requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos based on their metadata."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.text}")
            return {
                "success": False,
                "error": f"OpenAI API error: {response.status_code} - {response.text}"
            }
        
        # Extract the summary from the response
        response_data = response.json()
        summary = response_data["choices"][0]["message"]["content"].strip()
        
        return {
            "success": True,
            "summary": summary,
            "model": model
        }
        
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate summary: {str(e)}"
        }

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB limit

# Create download folder if not exists
if not os.path.exists(app.config['DOWNLOAD_FOLDER']):
    os.makedirs(app.config['DOWNLOAD_FOLDER'])

# Add custom headers to avoid 403 errors
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class DownloadTask:
    def __init__(self):
        self.progress = 0
        self.status = "pending"
        self.filename = ""
        self.created_at = datetime.now()
        self.file_path = ""
        self.file_size = 0
        self.download_speed = 0
        self.eta = 0

download_queue = {}
lock = threading.Lock()

# Clean up old downloads periodically
def cleanup_old_tasks():
    with lock:
        current_time = datetime.now()
        tasks_to_remove = []
        for task_id, task in download_queue.items():
            # Remove tasks older than 1 hour
            if (current_time - task.created_at).total_seconds() > 3600:
                tasks_to_remove.append(task_id)
                # Remove the file if it exists
                if task.file_path and os.path.exists(task.file_path):
                    try:
                        os.remove(task.file_path)
                    except Exception as e:
                        logger.error(f"Failed to remove file {task.file_path}: {str(e)}")
        
        for task_id in tasks_to_remove:
            del download_queue[task_id]

def sanitize_filename(title):
    # More thorough filename sanitization
    # Remove invalid characters and limit length
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Limit filename length to avoid path too long errors
    if len(sanitized) > 100:
        sanitized = sanitized[:97] + "..."
    return sanitized

def format_filesize(bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def get_video_info(url):
    """
    Get video information using multiple libraries with fallback mechanism.
    First tries pytube, then falls back to yt-dlp if available.
    """
    error_messages = []
    
    # First attempt: Try with pytube
    try:
        logger.info(f"Attempting to fetch video info with pytube: {url}")
        yt = YouTube(url, headers=headers)
        
        # Get all available streams
        all_streams = yt.streams.filter(progressive=True).order_by('resolution').desc()
        
        # If no progressive streams found, try getting audio and video streams separately
        if not all_streams:
            logger.info(f"No progressive streams found for {url}, trying separate audio/video streams")
            all_streams = yt.streams.filter(adaptive=True).order_by('resolution').desc()
        
        # Format stream information
        streams_info = [{
            'itag': stream.itag,
            'resolution': stream.resolution or 'Audio Only',
            'mime_type': stream.mime_type.split('/')[-1],
            'filesize': stream.filesize_approx,
            'filesize_formatted': format_filesize(stream.filesize_approx),
            'is_audio_only': stream.includes_audio_track and not stream.includes_video_track,
            'fps': getattr(stream, 'fps', 'N/A'),
            'source': 'pytube'
        } for stream in all_streams]
        
        return {
            'title': yt.title,
            'thumbnail': yt.thumbnail_url,
            'author': yt.author,
            'length': yt.length,
            'views': yt.views,
            'publish_date': yt.publish_date.strftime('%Y-%m-%d') if yt.publish_date else 'Unknown',
            'description': yt.description[:200] + '...' if yt.description and len(yt.description) > 200 else yt.description,
            'streams': streams_info,
            'source': 'pytube'
        }
    except Exception as e:
        error_message = f"Pytube error: {str(e)}"
        logger.warning(error_message)
        error_messages.append(error_message)
        
        # If pytube fails and yt-dlp is available, try with yt-dlp
        if YTDLP_AVAILABLE:
            try:
                logger.info(f"Attempting to fetch video info with yt-dlp: {url}")
                return get_video_info_ytdlp(url)
            except Exception as e2:
                error_message = f"yt-dlp error: {str(e2)}"
                logger.error(error_message)
                error_messages.append(error_message)
        
        # If both methods fail, try using yt-dlp as a subprocess
        try:
            logger.info(f"Attempting to fetch video info with yt-dlp subprocess: {url}")
            return get_video_info_ytdlp_subprocess(url)
        except Exception as e3:
            error_message = f"yt-dlp subprocess error: {str(e3)}"
            logger.error(error_message)
            error_messages.append(error_message)
    
    # If all methods fail, raise an exception with all error messages
    raise PytubeError(f"Failed to fetch video information using all available methods: {'; '.join(error_messages)}")

def get_video_info_ytdlp(url):
    """Get video information using yt-dlp library"""
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
        'force_generic_extractor': False
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # Get available formats
        ydl_opts['listformats'] = True
        with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
            formats_info = ydl2.extract_info(url, download=False)
            formats = formats_info.get('formats', [])
        
        # Format stream information
        streams_info = []
        for fmt in formats:
            if fmt.get('vcodec', 'none') != 'none' or fmt.get('acodec', 'none') != 'none':
                # Calculate approximate filesize if not provided
                filesize = fmt.get('filesize', 0)
                if not filesize and fmt.get('tbr'):
                    # Estimate filesize based on bitrate and duration
                    duration = info.get('duration', 0)
                    filesize = int((fmt.get('tbr', 0) * 1024 / 8) * duration)
                
                resolution = 'Audio Only'
                if fmt.get('height'):
                    resolution = f"{fmt.get('height')}p"
                
                streams_info.append({
                    'itag': fmt.get('format_id', ''),
                    'resolution': resolution,
                    'mime_type': fmt.get('ext', ''),
                    'filesize': filesize,
                    'filesize_formatted': format_filesize(filesize),
                    'is_audio_only': fmt.get('vcodec') == 'none' and fmt.get('acodec') != 'none',
                    'fps': fmt.get('fps', 'N/A'),
                    'source': 'yt-dlp'
                })
        
        # Sort streams by resolution (highest first)
        streams_info.sort(key=lambda x: (0 if x['is_audio_only'] else 1, 
                                        0 if x['resolution'] == 'Audio Only' else 
                                        int(x['resolution'].replace('p', ''))), 
                          reverse=True)
        
        return {
            'title': info.get('title', 'Unknown Title'),
            'thumbnail': info.get('thumbnail', ''),
            'author': info.get('uploader', 'Unknown'),
            'length': info.get('duration', 0),
            'views': info.get('view_count', 0),
            'publish_date': info.get('upload_date', 'Unknown'),
            'description': info.get('description', '')[:200] + '...' if info.get('description', '') and len(info.get('description', '')) > 200 else info.get('description', ''),
            'streams': streams_info,
            'source': 'yt-dlp'
        }

def get_video_info_ytdlp_subprocess(url):
    """Get video information using yt-dlp as a subprocess (most reliable method)"""
    # Create a temporary directory for the JSON output
    temp_dir = tempfile.mkdtemp()
    json_file = os.path.join(temp_dir, 'video_info.json')
    
    try:
        # Run yt-dlp to get video info and formats
        cmd = ['yt-dlp', '--dump-json', '--no-playlist', url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Get formats
        cmd_formats = ['yt-dlp', '-F', url]
        formats_result = subprocess.run(cmd_formats, capture_output=True, text=True)
        formats_output = formats_result.stdout
        
        # Parse formats output
        streams_info = []
        for line in formats_output.split('\n'):
            if line and not line.startswith('[') and not line.startswith('ID'):
                parts = line.split()
                if len(parts) >= 3:
                    format_id = parts[0]
                    
                    # Skip formats that are not useful
                    if format_id in ['sb0', 'sb1', 'sb2']:
                        continue
                        
                    # Extract resolution and codec info
                    resolution = 'Audio Only'
                    is_audio_only = False
                    mime_type = 'mp4'
                    
                    if 'audio only' in line.lower():
                        is_audio_only = True
                        mime_type = 'mp3'
                    else:
                        # Try to extract resolution
                        for part in parts:
                            if 'x' in part and part[0].isdigit():
                                width, height = part.split('x')
                                resolution = f"{height}p"
                                break
                    
                    # Extract file extension
                    for part in parts:
                        if part in ['mp4', 'webm', 'mp3', 'm4a', 'ogg']:
                            mime_type = part
                            break
                    
                    # Estimate filesize based on bitrate and duration
                    filesize = 0
                    for part in parts:
                        if part.endswith('KiB') or part.endswith('MiB'):
                            try:
                                if part.endswith('KiB'):
                                    filesize = float(part.replace('KiB', '')) * 1024
                                elif part.endswith('MiB'):
                                    filesize = float(part.replace('MiB', '')) * 1024 * 1024
                            except:
                                pass
                    
                    streams_info.append({
                        'itag': format_id,
                        'resolution': resolution,
                        'mime_type': mime_type,
                        'filesize': filesize,
                        'filesize_formatted': format_filesize(filesize),
                        'is_audio_only': is_audio_only,
                        'fps': 'N/A',
                        'source': 'yt-dlp-subprocess'
                    })
        
        # Sort streams by resolution (highest first)
        streams_info.sort(key=lambda x: (0 if x['is_audio_only'] else 1, 
                                        0 if x['resolution'] == 'Audio Only' else 
                                        int(x['resolution'].replace('p', ''))), 
                          reverse=True)
        
        return {
            'title': info.get('title', 'Unknown Title'),
            'thumbnail': info.get('thumbnail', ''),
            'author': info.get('uploader', 'Unknown'),
            'length': info.get('duration', 0),
            'views': info.get('view_count', 0),
            'publish_date': info.get('upload_date', 'Unknown'),
            'description': info.get('description', '')[:200] + '...' if info.get('description', '') and len(info.get('description', '')) > 200 else info.get('description', ''),
            'streams': streams_info,
            'source': 'yt-dlp-subprocess'
        }
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        search_query = request.form.get('search_query')
        
        # Handle YouTube search
        if search_query:
            try:
                return redirect(url_for('search_results', query=search_query))
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return render_template('error.html', error=f"Search error: {str(e)}")
        
        # Handle URL processing
        if not url:
            return render_template('error.html', error="Please enter a YouTube URL")
            
        # Validate URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return render_template('error.html', error="Invalid YouTube URL. Only YouTube links are supported.")
        
        try:
            if 'playlist' in url or 'list=' in url:
                return redirect(url_for('process_playlist', url=url))
            return redirect(url_for('select_quality', url=url))
        except Exception as e:
            logger.error(f"URL processing error: {str(e)}")
            return render_template('error.html', error=str(e))
    
    # Clean up old downloads
    cleanup_old_tasks()
    return render_template('index.html')

@app.route('/search')
def search_results():
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('index'))
    
    try:
        search_results = []
        s = Search(query)
        for result in s.results[:10]:  # Limit to 10 results
            search_results.append({
                'title': result.title,
                'thumbnail': result.thumbnail_url,
                'url': f"https://youtube.com/watch?v={result.video_id}",
                'author': result.author,
                'publish_date': result.publish_date.strftime('%Y-%m-%d') if result.publish_date else 'Unknown',
                'views': result.views
            })
        
        return render_template('search.html', results=search_results, query=query)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return render_template('error.html', error=f"Search error: {str(e)}")

@app.route('/select_quality')
def select_quality():
    url = request.args.get('url')
    if not url:
        return redirect(url_for('index'))
        
    try:
        video_info = get_video_info(url)
        return render_template('quality.html', 
                             video=video_info,
                             url=url)
    except PytubeError as e:
        logger.error(f"PytubeError: {str(e)}")
        return render_template('error.html', error=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return render_template('error.html', error=f"An unexpected error occurred: {str(e)}")

@app.route('/download', methods=['POST'])
def download():
    url = request.form.get('url')
    itag = request.form.get('itag')
    source = request.form.get('source', 'auto')  # Source can be 'pytube', 'yt-dlp', 'yt-dlp-subprocess' or 'auto'
    
    if not url or not itag:
        return render_template('error.html', error="Missing URL or stream selection")
    
    task_id = f"{time.time()}-{os.urandom(4).hex()}"
    
    def download_task():
        try:
            # Initialize task
            with lock:
                download_queue[task_id] = DownloadTask()
                download_queue[task_id].status = "Initializing download..."
            
            # Try to get video info to determine the best download method
            if source == 'auto':
                try:
                    video_info = get_video_info(url)
                    actual_source = video_info.get('source', 'pytube')
                except Exception as e:
                    logger.error(f"Error getting video info: {str(e)}")
                    actual_source = 'yt-dlp-subprocess'  # Fallback to most reliable method
            else:
                actual_source = source
            
            logger.info(f"Using {actual_source} for download")
            
            # Download based on the source
            if actual_source == 'pytube':
                download_with_pytube(url, itag, task_id)
            elif actual_source == 'yt-dlp' and YTDLP_AVAILABLE:
                download_with_ytdlp(url, itag, task_id)
            else:
                download_with_ytdlp_subprocess(url, itag, task_id)
                
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            with lock:
                download_queue[task_id].status = f"error: {str(e)}"
    
    # Start the download in a separate thread
    thread = threading.Thread(target=download_task)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('download_status', task_id=task_id))

def download_with_pytube(url, itag, task_id):
    """Download video using pytube library"""
    try:
        # Progress callback function for pytube
        def progress_callback(stream, chunk, bytes_remaining):
            with lock:
                if task_id in download_queue:
                    total_size = stream.filesize
                    bytes_downloaded = total_size - bytes_remaining
                    download_queue[task_id].progress = (bytes_downloaded / total_size) * 100
                    
                    # Calculate ETA and download speed
                    if bytes_downloaded > 0:
                        elapsed_time = time.time() - download_queue[task_id].created_at.timestamp()
                        if elapsed_time > 0:
                            download_speed = bytes_downloaded / elapsed_time
                            download_queue[task_id].download_speed = download_speed
                            
                            if download_speed > 0:
                                eta = bytes_remaining / download_speed
                                download_queue[task_id].eta = eta
        
        yt = YouTube(url, headers=headers, on_progress_callback=progress_callback)
        stream = yt.streams.get_by_itag(itag)
        
        if not stream:
            with lock:
                download_queue[task_id].status = "error: Selected stream not available"
            return
            
        filename = f"{sanitize_filename(yt.title)}.{stream.subtype}"
        filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        
        with lock:
            download_queue[task_id].filename = filename
            download_queue[task_id].file_path = filepath
            download_queue[task_id].file_size = stream.filesize
            download_queue[task_id].status = "Downloading with pytube..."
        
        # Download the file
        start_time = time.time()
        stream.download(output_path=app.config['DOWNLOAD_FOLDER'], filename=filename)
        download_time = time.time() - start_time
        
        with lock:
            download_queue[task_id].status = "complete"
            # Calculate average download speed
            if download_time > 0:
                download_queue[task_id].download_speed = stream.filesize / download_time
        
        logger.info(f"Download completed with pytube: {filename}")
        
    except Exception as e:
        logger.error(f"Pytube download error: {str(e)}")
        with lock:
            download_queue[task_id].status = f"error: Pytube download failed: {str(e)}"
        
        # Try fallback to yt-dlp
        if YTDLP_AVAILABLE:
            logger.info(f"Falling back to yt-dlp for download")
            download_with_ytdlp(url, itag, task_id)
        else:
            logger.info(f"Falling back to yt-dlp subprocess for download")
            download_with_ytdlp_subprocess(url, itag, task_id)

def download_with_ytdlp(url, itag, task_id):
    """Download video using yt-dlp library"""
    try:
        # Progress hook for yt-dlp
        def progress_hook(d):
            if d['status'] == 'downloading':
                with lock:
                    if task_id in download_queue:
                        # Update status
                        download_queue[task_id].status = f"Downloading with yt-dlp..."
                        
                        # Update progress
                        if 'total_bytes' in d and d['total_bytes'] > 0:
                            download_queue[task_id].file_size = d['total_bytes']
                            bytes_downloaded = d.get('downloaded_bytes', 0)
                            download_queue[task_id].progress = (bytes_downloaded / d['total_bytes']) * 100
                        elif 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                            download_queue[task_id].file_size = d['total_bytes_estimate']
                            bytes_downloaded = d.get('downloaded_bytes', 0)
                            download_queue[task_id].progress = (bytes_downloaded / d['total_bytes_estimate']) * 100
                        
                        # Update speed
                        if 'speed' in d and d['speed']:
                            download_queue[task_id].download_speed = d['speed']
                        
                        # Update ETA
                        if 'eta' in d and d['eta']:
                            download_queue[task_id].eta = d['eta']
            
            elif d['status'] == 'finished':
                with lock:
                    if task_id in download_queue:
                        download_queue[task_id].status = "Processing..."
        
        # Get video info to get the correct format
        ydl_opts = {
            'format': itag,
            'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': True
        }
        
        with lock:
            download_queue[task_id].status = "Preparing download with yt-dlp..."
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get the downloaded file path
            if 'requested_downloads' in info and info['requested_downloads']:
                filepath = info['requested_downloads'][0]['filepath']
                filename = os.path.basename(filepath)
            else:
                # Fallback if we can't get the exact path
                filename = f"{sanitize_filename(info['title'])}.{info['ext']}"
                filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
            
            with lock:
                download_queue[task_id].filename = filename
                download_queue[task_id].file_path = filepath
                download_queue[task_id].status = "complete"
            
            logger.info(f"Download completed with yt-dlp: {filename}")
    
    except Exception as e:
        logger.error(f"yt-dlp download error: {str(e)}")
        with lock:
            download_queue[task_id].status = f"error: yt-dlp download failed: {str(e)}"
        
        # Try fallback to subprocess method
        logger.info(f"Falling back to yt-dlp subprocess for download")
        download_with_ytdlp_subprocess(url, itag, task_id)

def download_with_ytdlp_subprocess(url, itag, task_id):
    """Download video using yt-dlp as a subprocess (most reliable method)"""
    try:
        # Create a unique filename for this download
        temp_filename = f"download_{task_id}"
        output_template = os.path.join(app.config['DOWNLOAD_FOLDER'], temp_filename)
        
        with lock:
            download_queue[task_id].status = "Preparing download with yt-dlp..."
        
        # Start the download process
        cmd = [
            'yt-dlp', 
            '-f', itag,
            '-o', f"{output_template}.%(ext)s",
            '--newline',
            url
        ]
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor the download progress
        for line in iter(process.stdout.readline, ''):
            # Update progress based on output
            if '[download]' in line:
                try:
                    # Try to parse progress percentage
                    if '%' in line:
                        progress_str = line.split('%')[0].split()[-1]
                        progress = float(progress_str)
                        
                        with lock:
                            if task_id in download_queue:
                                download_queue[task_id].progress = progress
                    
                    # Try to parse download speed
                    if 'at' in line and '/s' in line:
                        speed_str = line.split('at')[1].split('/s')[0].strip()
                        
                        # Convert speed to bytes/s
                        speed_value = float(speed_str.split()[0])
                        speed_unit = speed_str.split()[1]
                        
                        if speed_unit == 'KiB':
                            speed_bytes = speed_value * 1024
                        elif speed_unit == 'MiB':
                            speed_bytes = speed_value * 1024 * 1024
                        else:
                            speed_bytes = speed_value
                        
                        with lock:
                            if task_id in download_queue:
                                download_queue[task_id].download_speed = speed_bytes
                    
                    # Try to parse ETA
                    if 'ETA' in line:
                        eta_str = line.split('ETA')[1].strip()
                        
                        # Convert ETA to seconds
                        if ':' in eta_str:
                            eta_parts = eta_str.split(':')
                            if len(eta_parts) == 2:  # MM:SS
                                eta_seconds = int(eta_parts[0]) * 60 + int(eta_parts[1])
                            elif len(eta_parts) == 3:  # HH:MM:SS
                                eta_seconds = int(eta_parts[0]) * 3600 + int(eta_parts[1]) * 60 + int(eta_parts[2])
                            else:
                                eta_seconds = 0
                            
                            with lock:
                                if task_id in download_queue:
                                    download_queue[task_id].eta = eta_seconds
                except:
                    pass
            
            # Update status
            with lock:
                if task_id in download_queue:
                    download_queue[task_id].status = "Downloading with yt-dlp..."
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"yt-dlp process exited with code {process.returncode}")
        
        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(app.config['DOWNLOAD_FOLDER']) 
                           if f.startswith(temp_filename) and os.path.isfile(os.path.join(app.config['DOWNLOAD_FOLDER'], f))]
        
        if not downloaded_files:
            raise Exception("Download completed but file not found")
        
        # Get the first matching file
        downloaded_file = downloaded_files[0]
        
        # Get video info to get a proper title
        info_cmd = ['yt-dlp', '--dump-json', '--no-playlist', url]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True)
        
        if info_result.returncode == 0:
            try:
                info = json.loads(info_result.stdout)
                video_title = sanitize_filename(info.get('title', 'Unknown Title'))
                ext = os.path.splitext(downloaded_file)[1]
                
                # Rename the file to include the video title
                new_filename = f"{video_title}{ext}"
                new_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], new_filename)
                
                # Rename the file
                os.rename(os.path.join(app.config['DOWNLOAD_FOLDER'], downloaded_file), new_filepath)
                
                with lock:
                    download_queue[task_id].filename = new_filename
                    download_queue[task_id].file_path = new_filepath
                    download_queue[task_id].status = "complete"
                
                logger.info(f"Download completed with yt-dlp subprocess: {new_filename}")
            except Exception as e:
                logger.error(f"Error renaming file: {str(e)}")
                
                # Use the original filename if renaming fails
                with lock:
                    download_queue[task_id].filename = downloaded_file
                    download_queue[task_id].file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], downloaded_file)
                    download_queue[task_id].status = "complete"
        else:
            # Use the original filename if getting info fails
            with lock:
                download_queue[task_id].filename = downloaded_file
                download_queue[task_id].file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], downloaded_file)
                download_queue[task_id].status = "complete"
    
    except Exception as e:
        logger.error(f"yt-dlp subprocess download error: {str(e)}")
        with lock:
            download_queue[task_id].status = f"error: All download methods failed: {str(e)}"
    
    # Initialize the download task
    with lock:
        download_queue[task_id] = DownloadTask()
    
    # Start the download in a separate thread
    thread = threading.Thread(target=download_task)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('download_status', task_id=task_id))

@app.route('/status/<task_id>')
def download_status(task_id):
    with lock:
        task = download_queue.get(task_id, None)
    
    if not task:
        return render_template('error.html', error="Invalid or expired download task")
    
    if task.status == "complete":
        if os.path.exists(task.file_path):
            return send_file(task.file_path, as_attachment=True, download_name=task.filename)
        else:
            return render_template('error.html', error="Download file not found. It may have been deleted.")
    
    # Format download speed and ETA for display
    download_speed_formatted = format_filesize(task.download_speed) + "/s" if task.download_speed > 0 else "Calculating..."
    
    eta_formatted = ""
    if task.eta > 0:
        if task.eta < 60:
            eta_formatted = f"{int(task.eta)} seconds"
        elif task.eta < 3600:
            eta_formatted = f"{int(task.eta / 60)} minutes"
        else:
            eta_formatted = f"{int(task.eta / 3600)} hours, {int((task.eta % 3600) / 60)} minutes"
    
    return render_template('status.html', 
                          task=task, 
                          download_speed=download_speed_formatted,
                          eta=eta_formatted)

@app.route('/process_playlist')
def process_playlist():
    url = request.args.get('url')
    if not url:
        return redirect(url_for('index'))
        
    try:
        playlist = Playlist(url, headers=headers)
        
        # Get the first few videos for preview
        preview_videos = []
        for video_url in playlist.video_urls[:5]:  # Show first 5 videos
            try:
                yt = YouTube(video_url, headers=headers)
                preview_videos.append({
                    'title': yt.title,
                    'thumbnail': yt.thumbnail_url,
                    'author': yt.author,
                    'length': yt.length
                })
            except Exception:
                continue
        
        return render_template('playlist.html',
                             playlist_title=playlist.title,
                             playlist_url=url,
                             video_count=len(playlist.video_urls),
                             preview_videos=preview_videos)
    except Exception as e:
        logger.error(f"Playlist processing error: {str(e)}")
        return render_template('error.html', error=f"Failed to process playlist: {str(e)}")

@app.route('/download_playlist', methods=['POST'])
def download_playlist():
    url = request.form.get('url')
    quality = request.form.get('quality', 'highest')  # Default to highest quality
    
    if not url:
        return render_template('error.html', error="Missing playlist URL")
    
    task_id = f"{time.time()}-{os.urandom(4).hex()}"
    
    def playlist_task():
        try:
            with lock:
                download_queue[task_id] = DownloadTask()
                download_queue[task_id].status = "Initializing playlist download..."
            
            # Try with pytube first
            try:
                download_playlist_with_pytube(url, quality, task_id)
            except Exception as e:
                logger.error(f"Pytube playlist download error: {str(e)}")
                
                # Try with yt-dlp if pytube fails
                if YTDLP_AVAILABLE:
                    try:
                        download_playlist_with_ytdlp(url, quality, task_id)
                    except Exception as e2:
                        logger.error(f"yt-dlp playlist download error: {str(e2)}")
                        
                        # Try with yt-dlp subprocess as last resort
                        try:
                            download_playlist_with_ytdlp_subprocess(url, quality, task_id)
                        except Exception as e3:
                            logger.error(f"yt-dlp subprocess playlist download error: {str(e3)}")
                            with lock:
                                download_queue[task_id].status = f"error: All playlist download methods failed"
                else:
                    # Try with yt-dlp subprocess if yt-dlp library is not available
                    try:
                        download_playlist_with_ytdlp_subprocess(url, quality, task_id)
                    except Exception as e3:
                        logger.error(f"yt-dlp subprocess playlist download error: {str(e3)}")
                        with lock:
                            download_queue[task_id].status = f"error: All playlist download methods failed"
            
        except Exception as e:
            logger.error(f"Playlist download error: {str(e)}")
            with lock:
                download_queue[task_id].status = f"error: {str(e)}"
    
    # Start the download in a separate thread
    thread = threading.Thread(target=playlist_task)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('download_status', task_id=task_id))

def download_playlist_with_pytube(url, quality, task_id):
    """Download playlist using pytube library"""
    playlist = Playlist(url, headers=headers)
    zip_buffer = BytesIO()
    
    with lock:
        download_queue[task_id].filename = f"{sanitize_filename(playlist.title)}.zip"
        download_queue[task_id].status = "Preparing playlist download with pytube..."
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        total = len(playlist.video_urls)
        successful_downloads = 0
        failed_downloads = 0
        
        for i, video_url in enumerate(playlist.video_urls):
            try:
                yt = YouTube(video_url, headers=headers)
                
                # Select stream based on quality preference
                if quality == 'highest':
                    stream = yt.streams.get_highest_resolution()
                elif quality == 'lowest':
                    stream = yt.streams.get_lowest_resolution()
                elif quality == 'audio':
                    stream = yt.streams.get_audio_only()
                else:
                    stream = yt.streams.get_highest_resolution()
                
                if not stream:
                    logger.warning(f"No suitable stream found for {yt.title}")
                    continue
                    
                filename = f"{i+1:03d}. {sanitize_filename(yt.title)}.{stream.subtype}"
                filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
                
                # Update status with current video
                with lock:
                    download_queue[task_id].status = f"Downloading {i+1}/{total}: {yt.title}"
                
                # Download the video
                stream.download(output_path=app.config['DOWNLOAD_FOLDER'], filename=filename)
                
                # Add to zip file
                zip_file.write(filepath, filename)
                
                # Remove the individual file to save space
                os.remove(filepath)
                
                successful_downloads += 1
                
                # Update progress
                with lock:
                    download_queue[task_id].progress = (i+1)/total * 100
                    
            except Exception as e:
                logger.error(f"Error downloading video {video_url}: {str(e)}")
                failed_downloads += 1
                continue
    
    # Save the zip file
    zip_path = os.path.join(app.config['DOWNLOAD_FOLDER'], download_queue[task_id].filename)
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())
    
    with lock:
        download_queue[task_id].status = "complete"
        download_queue[task_id].file_path = zip_path
        
    logger.info(f"Playlist download completed with pytube: {successful_downloads} successful, {failed_downloads} failed")

def download_playlist_with_ytdlp(url, quality, task_id):
    """Download playlist using yt-dlp library"""
    # Determine format based on quality preference
    if quality == 'highest':
        format_spec = 'best'
    elif quality == 'lowest':
        format_spec = 'worst'
    elif quality == 'audio':
        format_spec = 'bestaudio'
    else:
        format_spec = 'best'
    
    # Create a temporary directory for downloads
    temp_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], f"playlist_temp_{task_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with lock:
            download_queue[task_id].status = "Preparing playlist download with yt-dlp..."
        
        # Progress hook for yt-dlp
        def progress_hook(d):
            if d['status'] == 'downloading':
                # Get playlist info
                playlist_info = d.get('info_dict', {})
                playlist_title = playlist_info.get('title', 'Playlist')
                
                # Get current entry info
                entries = playlist_info.get('entries', [])
                current_index = d.get('info_dict', {}).get('playlist_index', 0)
                total_entries = len(entries) if entries else playlist_info.get('playlist_count', 0)
                
                # Calculate overall progress
                if total_entries > 0:
                    # Entry progress (0-1)
                    entry_progress = 0
                    if 'total_bytes' in d and d['total_bytes'] > 0:
                        entry_progress = d.get('downloaded_bytes', 0) / d['total_bytes']
                    elif 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                        entry_progress = d.get('downloaded_bytes', 0) / d['total_bytes_estimate']
                    
                    # Overall progress (0-100)
                    overall_progress = ((current_index - 1) + entry_progress) / total_entries * 100
                    
                    with lock:
                        if task_id in download_queue:
                            download_queue[task_id].progress = overall_progress
                            download_queue[task_id].status = f"Downloading {current_index}/{total_entries}: {d.get('info_dict', {}).get('title', 'Video')}"
                            
                            # Update filename if not set
                            if not download_queue[task_id].filename:
                                download_queue[task_id].filename = f"{sanitize_filename(playlist_title)}.zip"
            
            elif d['status'] == 'finished' and d.get('info_dict', {}).get('_type', '') != 'playlist':
                # A single video has finished downloading
                with lock:
                    if task_id in download_queue:
                        download_queue[task_id].status = "Processing video..."
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': format_spec,
            'outtmpl': os.path.join(temp_dir, '%(playlist_index)s - %(title)s.%(ext)s'),
            'progress_hooks': [progress_hook],
            'quiet': False,
            'no_warnings': True,
            'ignoreerrors': True,  # Skip unavailable videos
        }
        
        # Download the playlist
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get playlist title
            playlist_title = info.get('title', 'Playlist')
            
            with lock:
                download_queue[task_id].status = "Creating ZIP file..."
                download_queue[task_id].filename = f"{sanitize_filename(playlist_title)}.zip"
            
            # Create ZIP file
            zip_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{sanitize_filename(playlist_title)}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add all downloaded files to the ZIP
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        zip_file.write(file_path, file)
            
            with lock:
                download_queue[task_id].status = "complete"
                download_queue[task_id].file_path = zip_path
            
            logger.info(f"Playlist download completed with yt-dlp: {playlist_title}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

def download_playlist_with_ytdlp_subprocess(url, quality, task_id):
    """Download playlist using yt-dlp as a subprocess"""
    # Determine format based on quality preference
    if quality == 'highest':
        format_spec = 'best'
    elif quality == 'lowest':
        format_spec = 'worst'
    elif quality == 'audio':
        format_spec = 'bestaudio'
    else:
        format_spec = 'best'
    
    # Create a temporary directory for downloads
    temp_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], f"playlist_temp_{task_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with lock:
            download_queue[task_id].status = "Preparing playlist download with yt-dlp..."
        
        # First, get playlist info
        info_cmd = ['yt-dlp', '--dump-json', '--flat-playlist', url]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True)
        
        playlist_title = "Playlist"
        total_videos = 0
        
        if info_result.returncode == 0:
            try:
                # Parse each line as a separate JSON object
                videos = [json.loads(line) for line in info_result.stdout.strip().split('\n') if line.strip()]
                total_videos = len(videos)
                
                # Try to get playlist title from the first video
                if videos and 'playlist_title' in videos[0]:
                    playlist_title = videos[0]['playlist_title']
                elif videos and 'playlist' in videos[0]:
                    playlist_title = videos[0]['playlist']
            except Exception as e:
                logger.error(f"Error parsing playlist info: {str(e)}")
        
        with lock:
            download_queue[task_id].filename = f"{sanitize_filename(playlist_title)}.zip"
        
        # Start the download process
        cmd = [
            'yt-dlp', 
            '-f', format_spec,
            '-o', os.path.join(temp_dir, '%(playlist_index)s - %(title)s.%(ext)s'),
            '--newline',
            url
        ]
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        current_video = 0
        
        # Monitor the download progress
        for line in iter(process.stdout.readline, ''):
            # Try to detect current video number
            if '[download]' in line and 'Downloading video' in line and 'of' in line:
                try:
                    parts = line.split('Downloading video')[1].split('of')[0].strip()
                    current_video = int(parts)
                    
                    # Update total if we have better information now
                    if 'of' in line:
                        total_part = line.split('of')[1].split()[0].strip()
                        if total_part.isdigit():
                            total_videos = int(total_part)
                except:
                    pass
            
            # Try to detect video title
            if '[download]' in line and 'Destination:' in line:
                try:
                    video_title = line.split('Destination:')[1].strip()
                    video_title = os.path.basename(video_title)
                    
                    with lock:
                        if task_id in download_queue:
                            download_queue[task_id].status = f"Downloading {current_video}/{total_videos}: {video_title}"
                except:
                    pass
            
            # Try to parse progress percentage
            if '[download]' in line and '%' in line:
                try:
                    progress_str = line.split('%')[0].split()[-1]
                    video_progress = float(progress_str)
                    
                    # Calculate overall progress
                    if total_videos > 0:
                        overall_progress = ((current_video - 1) + (video_progress / 100)) / total_videos * 100
                        
                        with lock:
                            if task_id in download_queue:
                                download_queue[task_id].progress = overall_progress
                except:
                    pass
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"yt-dlp process exited with code {process.returncode}")
        
        # Create ZIP file
        with lock:
            download_queue[task_id].status = "Creating ZIP file..."
        
        zip_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{sanitize_filename(playlist_title)}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all downloaded files to the ZIP
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, file)
        
        with lock:
            download_queue[task_id].status = "complete"
            download_queue[task_id].file_path = zip_path
        
        logger.info(f"Playlist download completed with yt-dlp subprocess: {playlist_title}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")
    
    # Start the download in a separate thread
    thread = threading.Thread(target=playlist_task)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('download_status', task_id=task_id))

@app.route('/api/task_status/<task_id>')
def api_task_status(task_id):
    """API endpoint to get task status for AJAX updates"""
    with lock:
        task = download_queue.get(task_id, None)
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify({
        'progress': task.progress,
        'status': task.status,
        'filename': task.filename,
        'download_speed': format_filesize(task.download_speed) + "/s" if task.download_speed > 0 else "Calculating...",
        'eta': task.eta,
        'is_complete': task.status == "complete"
    })

@app.route('/audio_only/<path:url>')
def audio_only(url):
    """Special route for audio-only downloads"""
    try:
        # Try with pytube first
        try:
            logger.info(f"Attempting to fetch audio streams with pytube: {url}")
            yt = YouTube(url, headers=headers)
            audio_streams = yt.streams.filter(only_audio=True).order_by('abr').desc()
            
            # Add source information
            for stream in audio_streams:
                stream.source = 'pytube'
            
            return render_template('audio.html',
                                  video_title=yt.title,
                                  video_author=yt.author,
                                  video_thumbnail=yt.thumbnail_url,
                                  audio_streams=audio_streams,
                                  url=url)
        except Exception as e:
            logger.warning(f"Pytube audio error: {str(e)}")
            
            # Try with yt-dlp if pytube fails
            if YTDLP_AVAILABLE:
                try:
                    logger.info(f"Attempting to fetch audio streams with yt-dlp: {url}")
                    return get_audio_ytdlp(url)
                except Exception as e2:
                    logger.error(f"yt-dlp audio error: {str(e2)}")
            
            # Try with yt-dlp subprocess as last resort
            logger.info(f"Attempting to fetch audio streams with yt-dlp subprocess: {url}")
            return get_audio_ytdlp_subprocess(url)
            
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return render_template('error.html', error=str(e))

def get_audio_ytdlp(url):
    """Get audio streams using yt-dlp library"""
    ydl_opts = {
        'format': 'bestaudio',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # Get available audio formats
        ydl_opts['listformats'] = True
        with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
            formats_info = ydl2.extract_info(url, download=False)
            formats = formats_info.get('formats', [])
        
        # Filter audio-only formats
        audio_formats = []
        for fmt in formats:
            if fmt.get('vcodec', 'none') == 'none' and fmt.get('acodec', 'none') != 'none':
                # Create a stream-like object compatible with the template
                audio_format = type('AudioStream', (), {})()
                audio_format.itag = fmt.get('format_id', '')
                audio_format.abr = fmt.get('abr', 0)
                if audio_format.abr:
                    audio_format.abr = f"{audio_format.abr}kbps"
                else:
                    audio_format.abr = "Unknown bitrate"
                
                audio_format.mime_type = f"audio/{fmt.get('ext', 'mp3')}"
                audio_format.filesize_approx = fmt.get('filesize', 0)
                if not audio_format.filesize_approx and fmt.get('tbr'):
                    # Estimate filesize based on bitrate and duration
                    duration = info.get('duration', 0)
                    audio_format.filesize_approx = int((fmt.get('tbr', 0) * 1024 / 8) * duration)
                
                audio_format.source = 'yt-dlp'
                audio_formats.append(audio_format)
        
        # Sort by bitrate (highest first)
        audio_formats.sort(key=lambda x: getattr(x, 'abr', 0), reverse=True)
        
        return render_template('audio.html',
                              video_title=info.get('title', 'Unknown Title'),
                              video_author=info.get('uploader', 'Unknown'),
                              video_thumbnail=info.get('thumbnail', ''),
                              audio_streams=audio_formats,
                              url=url)

def get_audio_ytdlp_subprocess(url):
    """Get audio streams using yt-dlp as a subprocess"""
    # Run yt-dlp to get video info
    cmd = ['yt-dlp', '--dump-json', '--no-playlist', url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to get video info: {result.stderr}")
    
    info = json.loads(result.stdout)
    
    # Get available formats
    cmd_formats = ['yt-dlp', '-F', url]
    formats_result = subprocess.run(cmd_formats, capture_output=True, text=True)
    
    if formats_result.returncode != 0:
        raise Exception(f"Failed to get format info: {formats_result.stderr}")
    
    formats_output = formats_result.stdout
    
    # Parse formats output to find audio-only streams
    audio_formats = []
    for line in formats_output.split('\n'):
        if line and 'audio only' in line.lower():
            parts = line.split()
            if len(parts) >= 3:
                format_id = parts[0]
                
                # Create a stream-like object compatible with the template
                audio_format = type('AudioStream', (), {})()
                audio_format.itag = format_id
                
                # Try to extract bitrate
                bitrate = "Unknown bitrate"
                for part in parts:
                    if 'k' in part and part[0].isdigit():
                        bitrate = part
                        break
                
                audio_format.abr = bitrate
                
                # Extract file extension
                mime_type = 'mp3'
                for part in parts:
                    if part in ['mp3', 'm4a', 'ogg', 'opus', 'webm']:
                        mime_type = part
                        break
                
                audio_format.mime_type = f"audio/{mime_type}"
                
                # Estimate filesize
                filesize = 0
                for part in parts:
                    if part.endswith('KiB') or part.endswith('MiB'):
                        try:
                            if part.endswith('KiB'):
                                filesize = float(part.replace('KiB', '')) * 1024
                            elif part.endswith('MiB'):
                                filesize = float(part.replace('MiB', '')) * 1024 * 1024
                        except:
                            pass
                
                audio_format.filesize_approx = filesize
                audio_format.source = 'yt-dlp-subprocess'
                audio_formats.append(audio_format)
    
    # Sort by format ID (usually higher is better quality)
    audio_formats.sort(key=lambda x: x.itag, reverse=True)
    
    return render_template('audio.html',
                          video_title=info.get('title', 'Unknown Title'),
                          video_author=info.get('uploader', 'Unknown'),
                          video_thumbnail=info.get('thumbnail', ''),
                          audio_streams=audio_formats,
                          url=url)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html', error="Internal server error"), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template('error.html', error="File too large. Maximum size is 1GB."), 413

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """API endpoint to get the status of a download task"""
    with lock:
        if task_id not in download_queue:
            return jsonify({
                'status': 'error: Task not found',
                'progress': 0
            })
        
        task = download_queue[task_id]
        
        # Format download speed
        download_speed = "N/A"
        if task.download_speed > 0:
            download_speed = format_filesize(task.download_speed) + "/s"
        
        # Format ETA
        eta = None
        if task.eta > 0:
            minutes, seconds = divmod(int(task.eta), 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                eta = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                eta = f"{minutes}m {seconds}s"
            else:
                eta = f"{seconds}s"
        
        return jsonify({
            'status': task.status,
            'progress': task.progress,
            'filename': task.filename,
            'download_speed': download_speed,
            'eta': eta,
            'is_complete': task.status == "complete"
        })

@app.route('/api/summarize', methods=['POST'])
def summarize_video():
    """API endpoint to generate an AI summary of a video"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'Missing video URL'
            })
        
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            return jsonify({
                'success': False,
                'error': 'OpenAI API key not configured. Please add your API key to the .env file.'
            })
        
        # Get video info
        try:
            video_info = get_video_info(url)
        except Exception as e:
            logger.error(f"Error getting video info for summary: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to get video information: {str(e)}"
            })
        
        # Generate summary
        summary_result = generate_video_summary(video_info)
        
        return jsonify(summary_result)
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

if __name__ == '__main__':
    # Schedule cleanup task
    cleanup_thread = threading.Thread(target=lambda: (time.sleep(3600), cleanup_old_tasks()))
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)