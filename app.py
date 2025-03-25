from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
import os
import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from typing import Union, List, Dict
from dataset_checker import DatasetChecker
from auto_fixer import DatasetAutoFixer
from problem_analyzer import ProblemAnalyzer
from generator import DatasetGenerator
import config
import requests
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import shutil
import re
import uuid
import numpy as np
import time
import io
import base64

# กำหนดตัวแปรเพื่อตรวจสอบว่าสามารถนำเข้าไลบรารีได้หรือไม่
TRANSFORMERS_AVAILABLE = False
TENSORFLOW_AVAILABLE = False
VISION_MODELS_AVAILABLE = False
MISTRAL_AVAILABLE = False
WEBDATASET_AVAILABLE = False
PILLOW_AVAILABLE = False
SOUNDFILE_AVAILABLE = False
TORCH_AVAILABLE = False
OLLAMA_AVAILABLE = False

# นำเข้าไลบรารีแบบมีเงื่อนไข
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ PyTorch - ฟีเจอร์ที่ใช้ PyTorch จะไม่ทำงาน")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ TensorFlow หรือมีปัญหาในการโหลด - ฟีเจอร์บางอย่างอาจไม่ทำงาน")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers สามารถใช้งานได้")
    
    # ลองนำเข้าโมเดลวิชัน
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        VISION_MODELS_AVAILABLE = True
        print("✅ Vision Models สามารถใช้งานได้")
    except ImportError:
        print("❌ ไม่พบโมเดลวิชันหรือมีปัญหาในการโหลด - ฟีเจอร์การประมวลผลภาพจะไม่ทำงาน")
except ImportError:
    print("❌ ไม่พบ Transformers หรือมีปัญหาในการโหลด - ฟีเจอร์ AI จะไม่ทำงาน")

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
    print("✅ Mistral API สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ Mistral API - ฟีเจอร์ Mistral จะไม่ทำงาน")

try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("✅ Ollama สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ Ollama - ฟีเจอร์ Ollama จะไม่ทำงาน")

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
    print("✅ WebDataset สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ WebDataset - ฟีเจอร์ WebDataset จะไม่ทำงาน")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
    print("✅ Pillow สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ Pillow - ฟีเจอร์การประมวลผลภาพจะไม่ทำงาน")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
    print("✅ SoundFile สามารถใช้งานได้")
except ImportError:
    print("❌ ไม่พบ SoundFile - ฟีเจอร์การประมวลผลเสียงจะไม่ทำงาน")

# Load environment variables
load_dotenv()

# Add API keys
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Vision model configurations
VISION_MODEL_SETTINGS = {
    'local': {
        'model_path': os.path.join(config.LOCAL_MODEL_DIR, 'vision_model'),
        'max_new_tokens': 500,
        'temperature': 0.7,
    },
    'ollama': {
        'model': 'llava',
        'endpoint': config.OLLAMA_DEFAULT_ENDPOINT,
    },
    'mistral': {
        'model': 'mistral-large-vision',
    }
}

# Stability AI configurations
STABILITY_API_HOST = "https://api.stability.ai"
STABILITY_SETTINGS = {
    'engine_id': "stable-diffusion-xl-1024-v1-0",
    'height': 1024,
    'width': 1024,
    'cfg_scale': 7,
    'samples': 1,
}

# เพิ่ม system_prompts เป็นตัวแปรระดับโมดูล
system_prompts = {
    'text_classification': """สร้างชุดข้อมูลสำหรับการจำแนกประเภทข้อความ โดยมี:
    1. ข้อความ (text)
    2. ประเภท/ป้ายกำกับ (label)
    3. ความมั่นใจในการทำนาย (confidence)""",
    
    'token_classification': """สร้างชุดข้อมูลสำหรับการระบุ token ในข้อความ โดยมี:
    1. ข้อความ (text)
    2. ตำแหน่งเริ่มต้นและสิ้นสุดของ token (start, end)
    3. ประเภทของ token (entity_type)""",
    
    'question_answering': """สร้างชุดข้อมูลสำหรับการตอบคำถาม โดยมี:
    1. บริบท/เนื้อหา (context)
    2. คำถาม (question)
    3. คำตอบ (answer)
    4. ตำแหน่งคำตอบในบริบท (start_position, end_position)""",
    
    'translation': """สร้างชุดข้อมูลสำหรับการแปลภาษา โดยมี:
    1. ข้อความต้นฉบับ (source_text)
    2. ภาษาต้นฉบับ (source_lang)
    3. ข้อความแปล (target_text)
    4. ภาษาเป้าหมาย (target_lang)""",
    
    'summarization': """สร้างชุดข้อมูลสำหรับการสรุปความ โดยมี:
    1. เนื้อหาต้นฉบับ (document)
    2. บทสรุป (summary)
    3. ความยาวบทสรุป (summary_length)"""
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(config.CHECK_RESULTS_DIR, exist_ok=True)
os.makedirs(config.GENERATED_DATASET_DIR, exist_ok=True)
os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)


# Global variables for model configuration
model_type = config.DEFAULT_MODEL_TYPE
api_key = os.getenv('MISTRAL_API_KEY', '')
model_path = ''
endpoint = config.OLLAMA_DEFAULT_ENDPOINT
model_name = config.MISTRAL_MODEL

def load_dataset(file_path: str, analyze_images: bool = False, vision_model: str = 'local') -> Union[pd.DataFrame, Dict]:
    """Load dataset from various file formats into pandas DataFrame or dictionary
    
    Supported formats:
    - JSON
    - CSV
    - Parquet
    - Arrow
    - Text
    - Image folder
    - Sound folder
    - WebDataset
    
    Args:
        file_path: พาธของไฟล์หรือโฟลเดอร์
        analyze_images: เปิดใช้การวิเคราะห์ภาพอัตโนมัติ
        vision_model: โมเดลที่ใช้วิเคราะห์ภาพ ('local', 'ollama', 'mistral')
    """
    file_path = Path(file_path)
    
    if file_path.is_file():
        # Handle file formats
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return pd.DataFrame(json.load(f))
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif file_path.suffix == '.arrow':
            table = pa.ipc.open_file(file_path).read_all()
            return table.to_pandas()
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return pd.DataFrame({'text': f.readlines()})
    
    elif file_path.is_dir():
        # Handle directories
        if any(file_path.glob('*.jpg')) or any(file_path.glob('*.png')):
            # Image folder
            if not PILLOW_AVAILABLE:
                print("ไม่สามารถโหลดไฟล์ภาพได้เนื่องจากไม่มี Pillow")
                return pd.DataFrame({"error": ["ไม่สามารถโหลดไฟล์ภาพได้เนื่องจากไม่มี Pillow"]})
                
            images = []
            for img_path in file_path.glob('*.[jp][pn][g]'):
                try:
                    with Image.open(img_path) as img:
                        image_data = {
                            'path': str(img_path),
                            'size': img.size,
                            'mode': img.mode
                        }
                        
                        # วิเคราะห์ภาพถ้าเปิดใช้งาน
                        if analyze_images and VISION_MODELS_AVAILABLE:
                            try:
                                analysis = analyze_image_with_vision_model(str(img_path), vision_model)
                                image_data.update({
                                    'analysis': analysis['description'],
                                    'analysis_timestamp': analysis['timestamp']
                                })
                            except Exception as e:
                                print(f"Warning: Failed to analyze image {img_path}: {e}")
                        elif analyze_images and not VISION_MODELS_AVAILABLE:
                            print("ไม่สามารถวิเคราะห์ภาพได้เนื่องจากไม่มีโมเดลวิชัน")
                        
                        images.append(image_data)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            return pd.DataFrame(images)
            
        elif any(file_path.glob('*.wav')) or any(file_path.glob('*.mp3')):
            # Sound folder
            if not SOUNDFILE_AVAILABLE:
                print("ไม่สามารถโหลดไฟล์เสียงได้เนื่องจากไม่มี SoundFile")
                return pd.DataFrame({"error": ["ไม่สามารถโหลดไฟล์เสียงได้เนื่องจากไม่มี SoundFile"]})
                
            audio_files = []
            for audio_path in file_path.glob('*.[wm][ap][v3]'):
                try:
                    data, samplerate = sf.read(audio_path)
                    audio_files.append({
                        'path': str(audio_path),
                        'samplerate': samplerate,
                        'duration': len(data) / samplerate,
                        'channels': data.shape[1] if len(data.shape) > 1 else 1
                    })
                except Exception as e:
                    print(f"Error loading audio {audio_path}: {e}")
            return pd.DataFrame(audio_files)
    
    # Handle WebDataset
    if str(file_path).endswith('.tar'):
        if not WEBDATASET_AVAILABLE:
            print("ไม่สามารถโหลด WebDataset ได้เนื่องจากไม่มี webdataset library")
            return pd.DataFrame({"error": ["ไม่สามารถโหลด WebDataset ได้เนื่องจากไม่มี webdataset library"]})
            
        try:
            dataset = wds.WebDataset(str(file_path))
            samples = []
            for sample in dataset:
                samples.append(sample)
            return pd.DataFrame(samples)
        except Exception as e:
            print(f"Error loading WebDataset: {e}")
    
    raise ValueError(f"Unsupported file format or directory: {file_path}")

def save_dataset(data: Union[pd.DataFrame, Dict], output_path: str, format: str = None):
    """Save DataFrame or dictionary to various formats
    
    Supported formats:
    - JSON
    - CSV
    - Parquet
    - Arrow
    - Text
    """
    output_path = Path(output_path)
    
    # Determine format from file extension if not specified
    if format is None:
        format = output_path.suffix.lstrip('.')
    
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        data.to_json(output_path, orient='records', force_ascii=False, indent=2)
    elif format == 'csv':
        data.to_csv(output_path, index=False)
    elif format == 'parquet':
        data.to_parquet(output_path)
    elif format == 'arrow':
        table = pa.Table.from_pandas(data)
        with pa.ipc.new_file(output_path, table.schema) as writer:
            writer.write_table(table)
    elif format == 'txt':
        if 'text' in data.columns:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(data['text'])
        else:
            data.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    """อัปโหลดและตรวจสอบคุณภาพชุดข้อมูล"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # ตรวจสอบพารามิเตอร์เพิ่มเติม
    output_format = request.form.get('output_format', 'html')
    enable_img_analysis = request.form.get('enable_img_analysis', 'false').lower() == 'true'
    
    # ตรวจสอบว่ารูปแบบที่ร้องขอรองรับหรือไม่
    supported_output_formats = ['html', 'json', 'csv', 'text', 'pdf']
    if output_format not in supported_output_formats:
        return jsonify({
            'error': f'รูปแบบผลลัพธ์ไม่รองรับ รองรับเฉพาะ: {", ".join(supported_output_formats)}'
        }), 400
    
    # บันทึกไฟล์ upload
    file_path = os.path.join(config.UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)
    
    # ตรวจสอบว่าเป็นชุดข้อมูลภาพหรือไม่
    is_image_dataset = False
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension in ['.zip', '.tar'] or (os.path.isdir(file_path) and any(f.endswith(('.jpg', '.jpeg', '.png', '.gif')) for f in os.listdir(file_path))):
        is_image_dataset = True
    
    try:
        # โหลดและตรวจสอบชุดข้อมูล
        dataset = load_dataset(file_path, enable_img_analysis=enable_img_analysis and is_image_dataset)
        
        # สร้างชื่อไฟล์ผลลัพธ์
        timestamp = int(time.time())
        result_filename = f"check_results_{timestamp}"
        report_path = os.path.join(config.CHECK_RESULTS_DIR, f"{result_filename}.{output_format}")
        
        # ตรวจสอบคุณภาพ
        checker = DatasetChecker(dataset, name=os.path.splitext(file.filename)[0])
        report = checker.run_quality_check()
        
        # บันทึกผลลัพธ์ตามรูปแบบที่ร้องขอ
        if output_format == 'html':
            report.to_html(report_path)
        elif output_format == 'json':
            report.to_json(report_path)
        elif output_format == 'csv':
            report.to_csv(report_path)
        elif output_format == 'pdf':
            report.to_pdf(report_path)
        else:  # text
            report.to_text(report_path)
        
        # สร้าง response
        report_url = f"/download/{result_filename}.{output_format}"
        return jsonify({
            'message': 'Upload and check successful',
            'filename': file.filename,
            'quality_score': report.overall_score,
            'report_url': report_url,
            'output_format': output_format,
            'error_count': report.error_count,
            'warning_count': report.warning_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_dataset():
    dataset_path = request.json.get('dataset_path')
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': 'Invalid dataset path'}), 400
    
    analyzer = ProblemAnalyzer()
    analysis_results = analyzer.analyze_dataset(dataset_path)
    
    output_path = os.path.join(
        config.CHECK_RESULTS_DIR,
        f"analysis_{os.path.basename(dataset_path)}"
    )
    analyzer.save_analysis_results(analysis_results, output_path)
    
    return jsonify({
        'message': 'Analysis completed',
        'results_path': output_path
    })

@app.route('/fix', methods=['POST'])
def fix_dataset():
    dataset_path = request.json.get('dataset_path')
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': 'Invalid dataset path'}), 400
    
    fixer = DatasetAutoFixer()
    fix_results = fixer.fix_dataset(dataset_path)
    
    output_path = os.path.join(
        config.CHECK_RESULTS_DIR,
        f"fix_results_{os.path.basename(dataset_path)}"
    )
    fixer.save_fix_results(fix_results, output_path)
    
    return jsonify({
        'message': 'Fix completed',
        'results': fix_results,
        'results_path': output_path
    })

@app.route('/results/<path:filename>')
def get_results(filename):
    """ดาวน์โหลดไฟล์ผลลัพธ์หรือชุดข้อมูลที่บันทึก"""
    # ตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ check_results หรือ generated_dataset
    if os.path.exists(os.path.join(config.CHECK_RESULTS_DIR, filename)):
        return send_file(
            os.path.join(config.CHECK_RESULTS_DIR, filename),
            as_attachment=True
        )
    elif os.path.exists(os.path.join(config.GENERATED_DATASET_DIR, filename)):
        return send_file(
            os.path.join(config.GENERATED_DATASET_DIR, filename),
            as_attachment=True
        )
    else:
        return jsonify({'error': 'ไม่พบไฟล์ที่ต้องการ'}), 404

def generate_nlp_dataset(task_type, prompt, num_samples):
    """สร้างชุดข้อมูลสำหรับงาน NLP ประเภทต่างๆ"""
    try:
        if task_type not in system_prompts:
            raise ValueError(f"ไม่รองรับงานประเภท: {task_type}")
            
        full_prompt = f"{system_prompts[task_type]}\n\nสร้างชุดข้อมูลตามคำอธิบายต่อไปนี้: {prompt}"
        
        response = get_model_response(full_prompt)
        data = json.loads(response)  # แปลง response เป็น JSON
        
        return pd.DataFrame(data)
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการสร้างชุดข้อมูล NLP: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate_dataset():
    prompt = request.json.get('prompt')
    num_samples = int(request.json.get('num_samples', 10))
    task_type = request.json.get('task_type', '')
    output_format = request.json.get('output_format', 'json')
    report_format = request.json.get('report_format', 'html')
    
    # ตรวจสอบรูปแบบไฟล์ที่รองรับ
    supported_formats = ['json', 'csv', 'parquet', 'arrow', 'txt']
    if output_format not in supported_formats:
        return jsonify({
            'error': f'รูปแบบไฟล์ไม่รองรับ รองรับเฉพาะ: {", ".join(supported_formats)}'
        }), 400
    
    # ตรวจสอบรูปแบบรายงานที่รองรับ
    supported_report_formats = ['html', 'json', 'csv', 'pdf', 'text']
    if report_format not in supported_report_formats:
        return jsonify({
            'error': f'รูปแบบรายงานไม่รองรับ รองรับเฉพาะ: {", ".join(supported_report_formats)}'
        }), 400
    
    if not prompt:
        return jsonify({'error': 'กรุณาระบุ prompt'}), 400
    
    try:
        # ดึงคำแนะนำจากระบบเรียนรู้
        improvement_suggestions = error_learning_system.suggest_improvements(task_type)
        
        # ปรับปรุง prompt ตามคำแนะนำ
        enhanced_prompt = prompt
        if improvement_suggestions:
            enhanced_prompt = f"{prompt}\n\nคำแนะนำเพิ่มเติม:\n" + "\n".join(improvement_suggestions)
        
        # สร้างชุดข้อมูล
        if task_type:
            dataset = generate_nlp_dataset(task_type, enhanced_prompt, num_samples)
        else:
            generator = DatasetGenerator()
            dataset = generator.generate_dataset(enhanced_prompt, num_samples)
        
        # ใช้กฎการปรับปรุงที่เรียนรู้มา
        dataset = error_learning_system.apply_improvements(dataset)
        
        # ตรวจสอบคุณภาพ
        checker = DatasetChecker(dataset, name="generated_dataset")
        report = checker.run_quality_check()
        
        # บันทึกข้อผิดพลาดที่พบ (ถ้ามี)
        if report.error_count > 0:
            for error in report.errors:
                error_learning_system.add_error({
                    'type': error.get('type', 'unknown'),
                    'field': error.get('field'),
                    'message': error.get('message'),
                    'task_type': task_type,
                    'prompt': prompt
                })
        
        # สร้างชื่อไฟล์และบันทึก
        timestamp = int(time.time())
        filename = f"generated_dataset_{timestamp}.{output_format}"
        output_path = os.path.join(config.GENERATED_DATASET_DIR, filename)
        save_dataset(dataset, output_path, format=output_format)
        
        # บันทึกรายงานตามรูปแบบที่ร้องขอ
        report_filename = f"check_results_generated_{timestamp}.{report_format}"
        report_path = os.path.join(config.CHECK_RESULTS_DIR, report_filename)
        
        if report_format == 'html':
            report.to_html(report_path)
        elif report_format == 'json':
            report.to_json(report_path)
        elif report_format == 'csv':
            report.to_csv(report_path)
        elif report_format == 'pdf':
            report.to_pdf(report_path)
        else:  # text
            report.to_text(report_path)
        
        # ดึงสถิติข้อผิดพลาด
        error_stats = error_learning_system.get_error_statistics()
        
        return jsonify({
            'message': 'สร้างชุดข้อมูลสำเร็จ',
            'task_type': task_type,
            'output_format': output_format,
            'report_format': report_format,
            'quality_score': report.overall_score,
            'dataset_path': output_path,
            'dataset_url': f"/download/{filename}",
            'report_url': f"/download/{report_filename}",
            'supported_formats': supported_formats,
            'improvement_suggestions': improvement_suggestions,
            'error_statistics': error_stats
        })
        
    except Exception as e:
        # บันทึกข้อผิดพลาดที่เกิดขึ้น
        error_learning_system.add_error({
            'type': 'generation_error',
            'message': str(e),
            'task_type': task_type,
            'prompt': prompt
        })
        return jsonify({'error': str(e)}), 500

def get_model_response(prompt):
    """ดึงการตอบสนองจากโมเดล"""
    global model_type, api_key, model_path, endpoint, model_name
    
    # ตรวจสอบว่ามีไลบรารีที่จำเป็นหรือไม่
    if model_type == 'mistral' and not MISTRAL_AVAILABLE:
        return "ไม่สามารถใช้ Mistral API ได้เนื่องจากไม่มีไลบรารี mistralai"
        
    if model_type == 'ollama' and not OLLAMA_AVAILABLE:
        return "ไม่สามารถใช้ Ollama ได้เนื่องจากไม่มีไลบรารี ollama"
        
    if model_type == 'local' and (not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE):
        return "ไม่สามารถใช้โมเดล local ได้เนื่องจากไม่มีไลบรารี transformers หรือ torch"
    
    try:
        if model_type == 'mistral':
            client = Mistral(api_key=api_key)
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            chat_response = client.chat.complete(
                model=model_name,
                messages=messages,
                **config.MODEL_SETTINGS
            )
            return chat_response.choices[0].message.content
            
        elif model_type == 'local':
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, **config.MODEL_SETTINGS)
            return tokenizer.decode(outputs[0])
            
        elif model_type == 'ollama':
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    **config.MODEL_SETTINGS
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"Error: {response.status_code}"
        else:
            return "Unsupported model type"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/update-model-config', methods=['POST'])
def update_model_config():
    global model_type, api_key, model_path, endpoint, model_name
    
    data = request.json
    model_type = data.get('model_type', config.DEFAULT_MODEL_TYPE)
    
    if model_type == 'mistral':
        api_key = data.get('api_key', '')
        model_name = data.get('model_name', config.MISTRAL_MODEL)
    elif model_type == 'local':
        model_path = data.get('model_path', '')
    elif model_type == 'ollama':
        endpoint = data.get('endpoint', config.OLLAMA_DEFAULT_ENDPOINT)
        model_name = data.get('model_name', '')
        
    # Test connection with sample prompt
    test_response = get_model_response("Test connection")
    
    return jsonify({
        'status': 'success',
        'message': 'Model configuration updated',
        'test_response': test_response
    })

@app.route('/download-hf-model', methods=['POST'])
def download_hf_model():
    model_id = request.json.get('model_id')
    if not model_id:
        return jsonify({'error': 'Model ID is required'}), 400
        
    try:
        save_path = os.path.join(config.LOCAL_MODEL_DIR, model_id.split('/')[-1])
        os.makedirs(save_path, exist_ok=True)
        
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model downloaded to {save_path}',
            'model_path': save_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-progress/<model_id>')
def download_progress(model_id):
    def generate():
        total = 100
        for i in range(total + 1):
            yield f"data: {json.dumps({'progress': i})}\n\n"
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

def enhance_prompt_with_ai(prompt):
    """ใช้ AI Model ที่กำหนดเพื่อปรับปรุง prompt"""
    try:
        system_prompt = """คุณเป็นผู้เชี่ยวชาญในการสร้างชุดข้อมูล NLP
        หน้าที่ของคุณคือวิเคราะห์คำอธิบายชุดข้อมูลและเสนอแนะการปรับปรุงให้ครอบคลุมมากขึ้น
        โดยคำนึงถึง:
        1. โครงสร้างข้อมูล (เช่น ฟิลด์ที่จำเป็น, ความสัมพันธ์ระหว่างข้อมูล)
        2. ประเภทข้อมูล (เช่น ข้อความ, ตัวเลข, วันที่)
        3. ขอบเขตและข้อจำกัด (เช่น ความยาวข้อความ, ภาษาที่รองรับ)
        4. คุณภาพและความถูกต้อง (เช่น การตรวจสอบความถูกต้อง, ความสอดคล้อง)
        5. ความสมบูรณ์ของข้อมูล (เช่น ข้อมูลที่จำเป็นครบถ้วน)
        6. ประเภทงาน NLP (เช่น การจำแนก, การตอบคำถาม, การแปลภาษา)
        7. รูปแบบผลลัพธ์ที่ต้องการ (เช่น ป้ายกำกับ, คะแนน, ข้อความ)"""
        
        full_prompt = f"{system_prompt}\n\nปรับปรุงคำอธิบายชุดข้อมูลต่อไปนี้ให้ครอบคลุมมากขึ้น: {prompt}"
        
        response = get_model_response(full_prompt)
        
        # สร้างคำแนะนำ 3 แบบ
        suggestions = [
            response,
            response + "\nเพิ่มเติม: ระบุเงื่อนไขการตรวจสอบความถูกต้องและคุณภาพข้อมูล",
            response + "\nเพิ่มเติม: กำหนดรูปแบบและมาตรฐานข้อมูลสำหรับงาน NLP"
        ]
        
        # วิเคราะห์คุณลักษณะ
        features_prompt = """วิเคราะห์และระบุคุณลักษณะสำคัญในคำอธิบายชุดข้อมูล โดยแบ่งเป็น:
        1. ประเภทงาน NLP ที่เหมาะสม
        2. โครงสร้างข้อมูลที่จำเป็น
        3. รูปแบบผลลัพธ์ที่ต้องการ
        4. ข้อจำกัดและเงื่อนไขพิเศษ
        
        คำอธิบายชุดข้อมูล: """ + response
        
        features_response = get_model_response(features_prompt)
        features = features_response.split('\n')
        
        return {
            "suggestions": suggestions,
            "features": features,
            "nlp_tasks": list(system_prompts.keys())  # ส่งรายการงาน NLP ที่รองรับ
        }
    except Exception as e:
        raise Exception(f"Error enhancing prompt: {str(e)}")

@app.route('/enhance-prompt', methods=['POST'])
def enhance_prompt():
    """Endpoint สำหรับปรับปรุง prompt"""
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'กรุณาระบุ prompt'}), 400
    
    try:
        result = enhance_prompt_with_ai(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def enhance_dataset(dataset, task_type=None):
    """ปรับปรุงคุณภาพของชุดข้อมูล"""
    try:
        # ตรวจสอบและแก้ไขข้อมูลที่ขาดหาย
        dataset = dataset.fillna({
            'text': '',
            'label': 'unknown',
            'confidence': 0.0,
            'features': [],
            'metadata': {}
        })
        
        # ปรับปรุงข้อความ
        if 'text' in dataset.columns:
            dataset['text'] = dataset['text'].apply(lambda x: x.strip())
            dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x.split()))  # ลบช่องว่างซ้ำ
        
        # ปรับปรุงป้ายกำกับ
        if 'label' in dataset.columns:
            dataset['label'] = dataset['label'].str.lower()
            dataset['label'] = dataset['label'].str.strip()
        
        # เพิ่มความมั่นใจในการทำนาย
        if 'confidence' in dataset.columns:
            dataset['confidence'] = dataset['confidence'].clip(0, 1)
        
        # เพิ่มคุณลักษณะตามประเภทงาน
        if task_type:
            if task_type == 'text_classification':
                # เพิ่มความหลากหลายของข้อความ
                dataset['text_length'] = dataset['text'].str.len()
                dataset['word_count'] = dataset['text'].str.split().str.len()
                
            elif task_type == 'token_classification':
                # เพิ่มข้อมูลตำแหน่ง
                dataset['start_position'] = dataset['start_position'].fillna(0)
                dataset['end_position'] = dataset['end_position'].fillna(dataset['text'].str.len())
                
            elif task_type == 'question_answering':
                # เพิ่มความยาวบริบทและคำถาม
                dataset['context_length'] = dataset['context'].str.len()
                dataset['question_length'] = dataset['question'].str.len()
                
            elif task_type == 'translation':
                # เพิ่มข้อมูลภาษา
                dataset['source_lang'] = dataset['source_lang'].fillna('th')
                dataset['target_lang'] = dataset['target_lang'].fillna('en')
                
            elif task_type == 'summarization':
                # เพิ่มอัตราส่วนการบีบอัด
                dataset['compression_ratio'] = dataset['summary'].str.len() / dataset['document'].str.len()
        
        # เพิ่มข้อมูลเมตาดาต้า
        dataset['metadata'] = dataset['metadata'].apply(lambda x: {
            **x,
            'enhanced_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'enhancement_version': '1.0'
        })
        
        return dataset
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการปรับปรุงชุดข้อมูล: {str(e)}")

@app.route('/enhance-dataset', methods=['POST'])
def enhance_dataset_endpoint():
    """Endpoint สำหรับปรับปรุงชุดข้อมูล"""
    dataset_path = request.json.get('dataset_path')
    task_type = request.json.get('task_type')
    
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': 'ไม่พบไฟล์ชุดข้อมูล'}), 400
    
    try:
        # โหลดชุดข้อมูล
        dataset = load_dataset(dataset_path)
        
        # ปรับปรุงชุดข้อมูล
        enhanced_dataset = enhance_dataset(dataset, task_type)
        
        # บันทึกชุดข้อมูลที่ปรับปรุงแล้ว
        output_path = os.path.join(
            config.GENERATED_DATASET_DIR,
            f"enhanced_{os.path.basename(dataset_path)}"
        )
        save_dataset(enhanced_dataset, output_path)
        
        # ตรวจสอบคุณภาพ
        checker = DatasetChecker(enhanced_dataset, name="enhanced_dataset")
        report = checker.run_quality_check()
        
        # บันทึกผลการตรวจสอบ
        check_output_path = os.path.join(
            config.CHECK_RESULTS_DIR,
            f"check_results_enhanced_{os.path.basename(dataset_path)}"
        )
        report.to_json(check_output_path)
        report.to_html(check_output_path.replace('.json', '.html'))
        
        return jsonify({
            'message': 'ปรับปรุงชุดข้อมูลสำเร็จ',
            'quality_score': report.overall_score,
            'enhanced_dataset_path': output_path,
            'results_path': check_output_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-dataset', methods=['POST'])
def save_dataset_endpoint():
    """Endpoint สำหรับบันทึกชุดข้อมูลในรูปแบบต่างๆ"""
    try:
        dataset_path = request.json.get('dataset_path')
        output_format = request.json.get('output_format', 'json')
        output_filename = request.json.get('output_filename')
        
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': 'ไม่พบไฟล์ชุดข้อมูล'}), 400
            
        if not output_filename:
            return jsonify({'error': 'กรุณาระบุชื่อไฟล์ที่ต้องการบันทึก'}), 400
        
        # ตรวจสอบรูปแบบไฟล์ที่รองรับ
        supported_formats = ['json', 'csv', 'parquet', 'arrow', 'txt']
        if output_format not in supported_formats:
            return jsonify({
                'error': f'รูปแบบไฟล์ไม่รองรับ รองรับเฉพาะ: {", ".join(supported_formats)}'
            }), 400
        
        # โหลดชุดข้อมูล
        dataset = load_dataset(dataset_path)
        
        # สร้างชื่อไฟล์พร้อมนามสกุล
        if not output_filename.endswith(f'.{output_format}'):
            output_filename = f"{output_filename}.{output_format}"
        
        # กำหนดพาธสำหรับบันทึกไฟล์
        output_path = os.path.join(config.GENERATED_DATASET_DIR, output_filename)
        
        # บันทึกชุดข้อมูลในรูปแบบที่เลือก
        save_dataset(dataset, output_path, format=output_format)
        
        return jsonify({
            'message': 'บันทึกชุดข้อมูลสำเร็จ',
            'output_path': output_path,
            'format': output_format,
            'supported_formats': supported_formats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-supported-formats', methods=['GET'])
def get_supported_formats():
    """Endpoint สำหรับดึงรายการรูปแบบไฟล์ที่รองรับ"""
    supported_formats = {
        'data_formats': ['json', 'csv', 'parquet', 'arrow', 'txt'],
        'image_formats': ['jpg', 'png'],
        'audio_formats': ['wav', 'mp3'],
        'archive_formats': ['tar']
    }
    return jsonify(supported_formats)

@app.route('/list-datasets', methods=['GET'])
def list_datasets():
    """Endpoint สำหรับดึงรายการชุดข้อมูลที่มีอยู่"""
    try:
        # รวบรวมไฟล์จากทั้งโฟลเดอร์ uploads และ generated_dataset
        datasets = []
        
        # ตรวจสอบไฟล์ในโฟลเดอร์ uploads
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                datasets.append({
                    'name': file,
                    'path': file_path,
                    'type': 'uploaded',
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path)
                })
        
        # ตรวจสอบไฟล์ในโฟลเดอร์ generated_dataset
        for file in os.listdir(config.GENERATED_DATASET_DIR):
            file_path = os.path.join(config.GENERATED_DATASET_DIR, file)
            if os.path.isfile(file_path):
                datasets.append({
                    'name': file,
                    'path': file_path,
                    'type': 'generated',
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path)
                })
        
        # เรียงลำดับตามเวลาที่แก้ไขล่าสุด
        datasets.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'message': 'ดึงรายการชุดข้อมูลสำเร็จ',
            'datasets': datasets
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_image_with_vision_model(image_path: str, model_type: str = 'local') -> dict:
    """Analyze image using a vision model and return a description"""
    if not VISION_MODELS_AVAILABLE:
        return {
            "description": "ไม่สามารถวิเคราะห์ภาพได้เนื่องจากไม่มีโมเดลวิชัน",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "none",
            "error": "Vision models not available"
        }
        
    if not PILLOW_AVAILABLE:
        return {
            "description": "ไม่สามารถโหลดไฟล์ภาพได้เนื่องจากไม่มี Pillow",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "none",
            "error": "Pillow not available"
        }
        
    # สร้าง prompt สำหรับการวิเคราะห์
    system_prompt = """คุณเป็น AI ที่เชี่ยวชาญในการวิเคราะห์ภาพ โปรดอธิบายภาพนี้อย่างละเอียด รวมถึง:
    1. สิ่งที่เห็นในภาพ
    2. องค์ประกอบหลักของภาพ
    3. สี บรรยากาศ และอารมณ์ของภาพ
    4. คุณภาพของภาพ (ความคมชัด ความสว่าง ความสมดุล)
    5. ปัญหาที่อาจพบในภาพ (เช่น เบลอ underexposed มืดเกินไป ฯลฯ)
    """
    
    try:
        # โหลดภาพด้วย PIL
        image = Image.open(image_path)
        
        # เลือกโมเดลตามประเภท
        if model_type == 'mistral' and MISTRAL_AVAILABLE:
            if not os.environ.get('MISTRAL_API_KEY'):
                return {
                    "description": "ไม่พบ MISTRAL_API_KEY สำหรับใช้งาน Mistral Vision",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "mistral",
                    "error": "API key not found"
                }
                
            client = Mistral(api_key=os.environ.get('MISTRAL_API_KEY'))
            
            # แปลงภาพเป็น base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # ส่งไปยัง Mistral API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "วิเคราะห์ภาพนี้และให้รายละเอียด"},
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_str}"}
                ]}
            ]
            
            response = client.chat.complete(
                model="mistral-large-vision",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            description = response.choices[0].message.content
            
        elif model_type == 'ollama' and OLLAMA_AVAILABLE:
            # แปลงภาพเป็น base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # ส่งไปยัง Ollama API
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": "llava", 
                    "prompt": f"{system_prompt}\n\nวิเคราะห์ภาพนี้และให้รายละเอียด", 
                    "images": [img_str]
                }
            )
            
            if response.status_code == 200:
                description = response.json().get('response', 'ไม่สามารถวิเคราะห์ภาพได้')
            else:
                description = f"Error: {response.status_code} - {response.text}"
                
        elif model_type == 'local' and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            # ใช้โมเดล local จาก transformers
            model_path = VISION_MODEL_SETTINGS['local']['model_path']
            
            if not os.path.exists(model_path):
                return {
                    "description": f"ไม่พบโมเดลที่ {model_path}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "local",
                    "error": "Model not found"
                }
                
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path)
            
            inputs = processor(images=image, text=system_prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=VISION_MODEL_SETTINGS['local']['max_new_tokens'],
                temperature=VISION_MODEL_SETTINGS['local']['temperature']
            )
            
            description = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            return {
                "description": f"ไม่รองรับโมเดล {model_type} หรือไม่มีไลบรารีที่จำเป็น",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_type,
                "error": "Unsupported model type or missing libraries"
            }
            
        # สร้างผลลัพธ์
        result = {
            "description": description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_type
        }
        
        return result
        
    except Exception as e:
        return {
            "description": "เกิดข้อผิดพลาดในการวิเคราะห์ภาพ",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_type,
            "error": str(e)
        }

def enhance_image_with_stability(image_path: str, prompt: str = None) -> str:
    """ปรับปรุงคุณภาพภาพด้วย Stability AI
    
    Args:
        image_path: พาธของไฟล์ภาพ
        prompt: คำอธิบายเพิ่มเติมสำหรับการปรับปรุงภาพ
    
    Returns:
        str: พาธของไฟล์ภาพที่ปรับปรุงแล้ว
    """
    try:
        if not STABILITY_API_KEY:
            raise ValueError("ไม่พบ STABILITY_API_KEY กรุณาตั้งค่าในไฟล์ .env")
        
        # โหลดและเตรียมภาพ
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # สร้าง prompt ถ้าไม่ได้ระบุ
        if not prompt:
            prompt = "Enhance this image while preserving its original content and style."
        
        # เรียกใช้ Stability AI API
        response = requests.post(
            f"{STABILITY_API_HOST}/v1/generation/image-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json"
            },
            json={
                "init_image": image_base64,
                "prompt": prompt,
                **STABILITY_SETTINGS
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Stability AI API error: {response.text}")
        
        # บันทึกภาพที่ปรับปรุงแล้ว
        result = response.json()
        enhanced_image_data = base64.b64decode(result['artifacts'][0]['base64'])
        
        # สร้างชื่อไฟล์ใหม่
        output_path = os.path.join(
            config.GENERATED_DATASET_DIR,
            f"enhanced_{os.path.basename(image_path)}"
        )
        
        # บันทึกไฟล์
        with open(output_path, "wb") as f:
            f.write(enhanced_image_data)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการปรับปรุงภาพ: {str(e)}")

@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    """Endpoint สำหรับวิเคราะห์ภาพ"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'ไม่พบไฟล์ภาพ'}), 400
        
        image = request.files['image']
        model_type = request.form.get('model_type', 'local')
        
        # บันทึกภาพชั่วคราว
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(temp_path)
        
        # วิเคราะห์ภาพ
        analysis = analyze_image_with_vision_model(temp_path, model_type)
        
        return jsonify({
            'message': 'วิเคราะห์ภาพสำเร็จ',
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance-image', methods=['POST'])
def enhance_image_endpoint():
    """Endpoint สำหรับปรับปรุงคุณภาพภาพ"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'ไม่พบไฟล์ภาพ'}), 400
        
        image = request.files['image']
        prompt = request.form.get('prompt')
        
        # บันทึกภาพชั่วคราว
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(temp_path)
        
        # ปรับปรุงภาพ
        enhanced_path = enhance_image_with_stability(temp_path, prompt)
        
        # วิเคราะห์ภาพที่ปรับปรุงแล้ว
        analysis = analyze_image_with_vision_model(enhanced_path)
        
        return jsonify({
            'message': 'ปรับปรุงภาพสำเร็จ',
            'enhanced_image_path': enhanced_path,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class ErrorLearningSystem:
    """ระบบเรียนรู้จากข้อผิดพลาดและปรับปรุงการ generate ข้อมูล"""
    
    def __init__(self):
        self.error_history = []
        self.improvement_rules = {}
        self.error_patterns = {}
        self.error_clusters = {}  # เก็บกลุ่มข้อผิดพลาดที่คล้ายกัน
        self.suggestion_confidence = {}  # ความมั่นใจในคำแนะนำ
        
    def add_error(self, error_info: dict):
        """บันทึกข้อผิดพลาดและวิเคราะห์รูปแบบ"""
        self.error_history.append({
            **error_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # วิเคราะห์รูปแบบข้อผิดพลาด
        error_type = error_info.get('type', 'unknown')
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = {
                'count': 0,
                'examples': [],
                'solutions': set(),
                'frequency': {}  # บันทึกความถี่ของรูปแบบข้อผิดพลาด
            }
        
        self.error_patterns[error_type]['count'] += 1
        self.error_patterns[error_type]['examples'].append(error_info)
        
        # บันทึกความถี่ของรูปแบบข้อผิดพลาด
        field = error_info.get('field', 'general')
        if field not in self.error_patterns[error_type]['frequency']:
            self.error_patterns[error_type]['frequency'][field] = 0
        self.error_patterns[error_type]['frequency'][field] += 1
        
        # จัดกลุ่มข้อผิดพลาดที่คล้ายกัน (Clustering)
        self._cluster_similar_errors(error_type, error_info)
        
        # สร้างกฎการปรับปรุงอัตโนมัติ
        self._generate_improvement_rules(error_type, error_info)
        
        # ปรับปรุงความมั่นใจในคำแนะนำ
        self._update_suggestion_confidence(error_type, error_info)
    
    def _cluster_similar_errors(self, error_type, error_info):
        """จัดกลุ่มข้อผิดพลาดที่คล้ายกันด้วย Simple Clustering Algorithm"""
        if 'message' not in error_info:
            return
            
        message = error_info.get('message', '')
        # สร้าง cluster key จากคำสำคัญในข้อความ
        words = set(message.lower().split())
        key_words = {w for w in words if len(w) > 3}  # เลือกเฉพาะคำที่มีความยาวมากกว่า 3 ตัวอักษร
        
        # ค้นหา cluster ที่เหมาะสม
        best_match = None
        best_score = 0
        
        for cluster_id, cluster in self.error_clusters.items():
            cluster_keywords = cluster['keywords']
            # คำนวณความคล้ายคลึงด้วย Jaccard similarity
            similarity = len(key_words.intersection(cluster_keywords)) / len(key_words.union(cluster_keywords)) if key_words.union(cluster_keywords) else 0
            
            if similarity > 0.3 and similarity > best_score:  # ต้องมีความคล้ายคลึงอย่างน้อย 30%
                best_match = cluster_id
                best_score = similarity
        
        # ถ้าไม่พบ cluster ที่เหมาะสม ให้สร้างใหม่
        if best_match is None:
            cluster_id = f"cluster_{len(self.error_clusters) + 1}"
            self.error_clusters[cluster_id] = {
                'keywords': key_words,
                'examples': [error_info],
                'count': 1,
                'error_types': {error_type: 1}
            }
        else:
            # เพิ่มข้อมูลในกลุ่มที่มีอยู่แล้ว
            self.error_clusters[best_match]['examples'].append(error_info)
            self.error_clusters[best_match]['count'] += 1
            
            if error_type in self.error_clusters[best_match]['error_types']:
                self.error_clusters[best_match]['error_types'][error_type] += 1
            else:
                self.error_clusters[best_match]['error_types'][error_type] = 1
            
            # ปรับปรุง keywords
            self.error_clusters[best_match]['keywords'].update(key_words)
    
    def _generate_improvement_rules(self, error_type: str, error_info: dict):
        """สร้างกฎการปรับปรุงจากข้อผิดพลาด"""
        if error_type not in self.improvement_rules:
            self.improvement_rules[error_type] = []
        
        # สร้างกฎตามประเภทข้อผิดพลาด
        if error_type == 'missing_field':
            self.improvement_rules[error_type].append({
                'condition': lambda x: error_info['field'] not in x,
                'action': lambda x: {**x, error_info['field']: error_info.get('default_value', None)},
                'confidence': self._calculate_rule_confidence(error_type, error_info),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        elif error_type == 'invalid_format':
            self.improvement_rules[error_type].append({
                'condition': lambda x: not self._validate_format(x, error_info['field'], error_info['format']),
                'action': lambda x: self._fix_format(x, error_info['field'], error_info['format']),
                'confidence': self._calculate_rule_confidence(error_type, error_info),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        elif error_type == 'out_of_range':
            self.improvement_rules[error_type].append({
                'condition': lambda x: not self._validate_range(x, error_info['field'], error_info['range']),
                'action': lambda x: self._fix_range(x, error_info['field'], error_info['range']),
                'confidence': self._calculate_rule_confidence(error_type, error_info),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # อัลกอริทึมการเรียนรู้แบบ Decision Tree อย่างง่าย
        elif error_type == 'generation_error':
            message = error_info.get('message', '')
            if 'index' in message.lower() or 'key' in message.lower():
                # สร้างกฎสำหรับแก้ไขข้อผิดพลาดเกี่ยวกับ index หรือ key
                self.improvement_rules[error_type].append({
                    'condition': lambda x: isinstance(x, dict) and any(k.startswith('index') for k in x.keys()),
                    'action': lambda x: {k: v for k, v in x.items() if not k.startswith('index')},
                    'confidence': 0.7,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })
            elif 'type' in message.lower():
                # สร้างกฎสำหรับแก้ไขข้อผิดพลาดเกี่ยวกับประเภทข้อมูล
                self.improvement_rules[error_type].append({
                    'condition': lambda x: True,  # ใช้กับทุกข้อมูล
                    'action': lambda x: {k: (str(v) if not isinstance(v, (int, float, bool, list, dict)) else v) for k, v in x.items()},
                    'confidence': 0.6,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })
    
    def _calculate_rule_confidence(self, error_type, error_info):
        """คำนวณความมั่นใจของกฎด้วยอัลกอริทึมวิเคราะห์ความถี่"""
        # ถ้าข้อผิดพลาดที่พบบ่อย ความมั่นใจจะสูง
        if error_type in self.error_patterns:
            count = self.error_patterns[error_type]['count']
            if count > 10:
                return 0.9
            elif count > 5:
                return 0.8
            elif count > 3:
                return 0.7
            elif count > 1:
                return 0.6
            else:
                return 0.5
        return 0.5
    
    def _update_suggestion_confidence(self, error_type, error_info):
        """ปรับปรุงความมั่นใจในคำแนะนำด้วยอัลกอริทึม Bayesian Update"""
        task_type = error_info.get('task_type', 'general')
        
        if task_type not in self.suggestion_confidence:
            self.suggestion_confidence[task_type] = {
                'error_types': {},
                'overall': 0.5  # ค่าเริ่มต้น
            }
        
        if error_type not in self.suggestion_confidence[task_type]['error_types']:
            self.suggestion_confidence[task_type]['error_types'][error_type] = 0.5  # ค่าเริ่มต้น
        
        # ปรับปรุงความมั่นใจด้วยการเรียนรู้แบบ Bayesian
        current_confidence = self.suggestion_confidence[task_type]['error_types'][error_type]
        
        # ถ้าข้อผิดพลาดเดิมยังคงเกิดขึ้น ลดความมั่นใจลง
        if self.error_patterns[error_type]['count'] > 1:
            new_confidence = current_confidence * 0.9
        else:
            # เพิ่มความมั่นใจเล็กน้อยเมื่อพบข้อผิดพลาดใหม่ (เพราะเราได้เรียนรู้เพิ่ม)
            new_confidence = current_confidence * 1.1
        
        # จำกัดค่าความมั่นใจให้อยู่ในช่วง [0.1, 0.95]
        new_confidence = max(0.1, min(0.95, new_confidence))
        
        self.suggestion_confidence[task_type]['error_types'][error_type] = new_confidence
        
        # ปรับปรุงความมั่นใจโดยรวม
        self.suggestion_confidence[task_type]['overall'] = sum(
            self.suggestion_confidence[task_type]['error_types'].values()
        ) / len(self.suggestion_confidence[task_type]['error_types'])
    
    def apply_improvements(self, data: Union[dict, pd.DataFrame]) -> Union[dict, pd.DataFrame]:
        """ใช้กฎการปรับปรุงกับข้อมูลด้วยอัลกอริทึมการเรียนรู้แบบใช้กฎ (Rule-based Learning)"""
        if isinstance(data, pd.DataFrame):
            improved_data = data.copy()
            
            # เรียงลำดับกฎตามความมั่นใจ (ใช้กฎที่มีความมั่นใจสูงก่อน)
            sorted_rules = []
            for error_type, rules in self.improvement_rules.items():
                for rule in rules:
                    sorted_rules.append((rule, error_type))
            
            sorted_rules.sort(key=lambda x: x[0].get('confidence', 0), reverse=True)
            
            # ใช้กฎตามลำดับความมั่นใจ
            for rule, error_type in sorted_rules:
                try:
                    mask = improved_data.apply(rule['condition'], axis=1)
                    improved_data.loc[mask] = improved_data.loc[mask].apply(rule['action'], axis=1)
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการใช้กฎ {error_type}: {str(e)}")
                    continue
                    
            return improved_data
        else:
            improved_data = data.copy()
            
            # เรียงลำดับกฎตามความมั่นใจ
            sorted_rules = []
            for error_type, rules in self.improvement_rules.items():
                for rule in rules:
                    sorted_rules.append((rule, error_type))
            
            sorted_rules.sort(key=lambda x: x[0].get('confidence', 0), reverse=True)
            
            # ใช้กฎตามลำดับความมั่นใจ
            for rule, error_type in sorted_rules:
                try:
                    if rule['condition'](improved_data):
                        improved_data = rule['action'](improved_data)
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการใช้กฎ {error_type}: {str(e)}")
                    continue
                    
            return improved_data
    
    def get_error_statistics(self) -> dict:
        """รายงานสถิติข้อผิดพลาดด้วยอัลกอริทึมวิเคราะห์แนวโน้ม (Trend Analysis)"""
        # วิเคราะห์แนวโน้มข้อผิดพลาดตามเวลา
        time_series = {}
        for error in self.error_history:
            date = error['timestamp'].split(' ')[0]  # เอาแค่วันที่
            if date not in time_series:
                time_series[date] = {'total': 0, 'types': {}}
            
            time_series[date]['total'] += 1
            
            error_type = error.get('type', 'unknown')
            if error_type not in time_series[date]['types']:
                time_series[date]['types'][error_type] = 0
            time_series[date]['types'][error_type] += 1
        
        # หาแนวโน้มการเปลี่ยนแปลง
        trend = {}
        dates = sorted(time_series.keys())
        if len(dates) > 1:
            first_date = dates[0]
            last_date = dates[-1]
            
            for error_type in self.error_patterns:
                first_count = time_series[first_date]['types'].get(error_type, 0)
                last_count = time_series[last_date]['types'].get(error_type, 0)
                
                if first_count > 0:
                    change = (last_count - first_count) / first_count * 100
                else:
                    change = 100 if last_count > 0 else 0
                
                trend[error_type] = {
                    'change_percent': change,
                    'direction': 'increasing' if change > 0 else 'decreasing' if change < 0 else 'stable'
                }
        
        return {
            'total_errors': len(self.error_history),
            'error_types': {
                error_type: {
                    'count': info['count'],
                    'percentage': (info['count'] / len(self.error_history)) * 100 if self.error_history else 0,
                    'recent_examples': info['examples'][-3:],  # แสดง 3 ตัวอย่างล่าสุด
                    'frequency_by_field': info.get('frequency', {})
                }
                for error_type, info in self.error_patterns.items()
            },
            'improvement_rules': {
                error_type: len(rules)
                for error_type, rules in self.improvement_rules.items()
            },
            'trends': trend,
            'time_series': time_series,
            'clusters': {
                cluster_id: {
                    'count': cluster['count'],
                    'error_types': cluster['error_types'],
                    'example': cluster['examples'][-1] if cluster['examples'] else None
                }
                for cluster_id, cluster in self.error_clusters.items()
            }
        }
    
    def suggest_improvements(self, task_type: str = None) -> List[str]:
        """แนะนำการปรับปรุงตามประวัติข้อผิดพลาดด้วยอัลกอริทึมการจัดอันดับ (Ranking Algorithm)"""
        suggestions = []
        
        # หาข้อผิดพลาดที่พบบ่อยและมีผลกระทบสูง
        ranked_errors = []
        for error_type, info in self.error_patterns.items():
            # คำนวณคะแนนผลกระทบ
            impact_score = info['count']
            
            # เพิ่มน้ำหนักตามความถี่ของข้อผิดพลาด
            if len(self.error_history) > 0:
                frequency = info['count'] / len(self.error_history)
                impact_score *= (1 + frequency)
            
            # เพิ่มน้ำหนักถ้าเป็นข้อผิดพลาดที่เกิดขึ้นในระยะเวลาล่าสุด
            if info['examples']:
                latest_error = info['examples'][-1]
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                try:
                    latest_time = latest_error.get('timestamp', '1970-01-01 00:00:00')
                    
                    # คำนวณความแตกต่างเวลาอย่างง่าย
                    if latest_time.split(' ')[0] == current_time.split(' ')[0]:  # เกิดในวันเดียวกัน
                        impact_score *= 1.5
                except:
                    pass
            
            ranked_errors.append((error_type, impact_score, info))
        
        # เรียงลำดับตามคะแนนผลกระทบ
        ranked_errors.sort(key=lambda x: x[1], reverse=True)
        
        # สร้างคำแนะนำจากข้อผิดพลาดที่จัดอันดับแล้ว
        for error_type, score, info in ranked_errors[:5]:  # แนะนำ 5 อันดับแรก
            if task_type:
                if task_type in system_prompts:
                    # ตรวจสอบว่าข้อผิดพลาดนี้เกี่ยวข้องกับ task_type นี้หรือไม่
                    task_relevant = False
                    for example in info['examples']:
                        if example.get('task_type') == task_type:
                            task_relevant = True
                            break
                    
                    if task_relevant:
                        suggestions.append(
                            f"สำหรับงาน {task_type}: ควรระวังข้อผิดพลาด {error_type} "
                            f"ที่พบ {info['count']} ครั้ง และมีความสำคัญสูง (คะแนน: {score:.2f})"
                        )
            else:
                suggestions.append(
                    f"ข้อผิดพลาด {error_type} พบบ่อยที่สุด ({info['count']} ครั้ง) "
                    f"และมีความสำคัญสูง (คะแนน: {score:.2f}) ควรตรวจสอบและป้องกัน"
                )
        
        # ถ้ามีข้อมูลจากการจัดกลุ่ม ให้เพิ่มคำแนะนำจากกลุ่มข้อผิดพลาด
        if self.error_clusters:
            # เรียงลำดับกลุ่มตามจำนวนข้อผิดพลาด
            ranked_clusters = sorted(
                self.error_clusters.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            
            for cluster_id, cluster in ranked_clusters[:2]:  # แนะนำจาก 2 กลุ่มแรก
                if cluster['examples']:
                    example = cluster['examples'][-1]
                    message = example.get('message', '')
                    
                    # สร้างคำแนะนำจากตัวอย่างข้อผิดพลาดในกลุ่ม
                    if message:
                        suggestions.append(
                            f"พบรูปแบบข้อผิดพลาดที่คล้ายกัน {cluster['count']} ครั้ง: "
                            f"'{message[:50]}...' ควรตรวจสอบและแก้ไข"
                        )
        
        return suggestions
    
    def get_suggestion_confidence(self, task_type: str = None) -> dict:
        """ดึงค่าความมั่นใจในคำแนะนำ"""
        if task_type and task_type in self.suggestion_confidence:
            return self.suggestion_confidence[task_type]
        elif 'general' in self.suggestion_confidence:
            return self.suggestion_confidence['general']
        else:
            return {'overall': 0.5, 'error_types': {}}
    
    def _validate_format(self, data: dict, field: str, format_spec: str) -> bool:
        """ตรวจสอบรูปแบบข้อมูล"""
        if field not in data:
            return False
        
        value = data[field]
        
        if format_spec == 'date':
            try:
                pd.to_datetime(value)
                return True
            except:
                return False
                
        elif format_spec == 'email':
            import re
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            return bool(re.match(email_pattern, str(value)))
            
        elif format_spec == 'number':
            return isinstance(value, (int, float))
            
        return True
    
    def _fix_format(self, data: dict, field: str, format_spec: str) -> dict:
        """แก้ไขรูปแบบข้อมูล"""
        fixed_data = data.copy()
        
        if field not in fixed_data:
            return fixed_data
            
        value = fixed_data[field]
        
        if format_spec == 'date':
            try:
                fixed_data[field] = pd.to_datetime(value).strftime('%Y-%m-%d')
            except:
                fixed_data[field] = None
                
        elif format_spec == 'email':
            if not isinstance(value, str):
                fixed_data[field] = None
            elif '@' not in value:
                fixed_data[field] = None
                
        elif format_spec == 'number':
            try:
                fixed_data[field] = float(value)
            except:
                fixed_data[field] = None
                
        return fixed_data
    
    def _validate_range(self, data: dict, field: str, range_spec: tuple) -> bool:
        """ตรวจสอบช่วงข้อมูล"""
        if field not in data:
            return False
            
        value = data[field]
        min_val, max_val = range_spec
        
        if isinstance(value, (int, float)):
            return min_val <= value <= max_val
            
        return True
    
    def _fix_range(self, data: dict, field: str, range_spec: tuple) -> dict:
        """แก้ไขช่วงข้อมูล"""
        fixed_data = data.copy()
        
        if field not in fixed_data:
            return fixed_data
            
        value = fixed_data[field]
        min_val, max_val = range_spec
        
        if isinstance(value, (int, float)):
            fixed_data[field] = max(min_val, min(value, max_val))
            
        return fixed_data
            
    def get_improvement_rules(self) -> dict:
        """ดึงกฎการปรับปรุงทั้งหมด"""
        # แปลงกฎให้เป็นรูปแบบที่อ่านง่าย
        readable_rules = {}
        for error_type, rules in self.improvement_rules.items():
            readable_rules[error_type] = []
            for rule in rules:
                readable_rules[error_type].append({
                    'confidence': rule.get('confidence', 0.5),
                    'created_at': rule.get('created_at', ''),
                    'description': f"กฎสำหรับปรับปรุงข้อผิดพลาดประเภท {error_type}"
                })
        
        return readable_rules
        
    def summarize_error_history(self) -> dict:
        """สรุปประวัติข้อผิดพลาดด้วยอัลกอริทึมวิเคราะห์สถิติ"""
        if not self.error_history:
            return {
                'total': 0,
                'message': 'ไม่มีประวัติข้อผิดพลาด'
            }
        
        # วิเคราะห์ประวัติข้อผิดพลาดตามประเภท
        error_types = {}
        for error in self.error_history:
            error_type = error.get('type', 'unknown')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # คำนวณสถิติ
        stats = {
            'total': len(self.error_history),
            'by_type': {},
            'most_common': None,
            'least_common': None,
            'recent_trend': None
        }
        
        if error_types:
            # หาประเภทที่พบบ่อยและน้อยที่สุด
            type_counts = {t: len(errors) for t, errors in error_types.items()}
            most_common = max(type_counts.items(), key=lambda x: x[1])
            least_common = min(type_counts.items(), key=lambda x: x[1])
            
            stats['most_common'] = {
                'type': most_common[0],
                'count': most_common[1],
                'percentage': (most_common[1] / len(self.error_history)) * 100
            }
            
            stats['least_common'] = {
                'type': least_common[0],
                'count': least_common[1],
                'percentage': (least_common[1] / len(self.error_history)) * 100
            }
            
            # วิเคราะห์แนวโน้มล่าสุด (10 รายการล่าสุด)
            recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
            recent_types = {}
            
            for error in recent_errors:
                error_type = error.get('type', 'unknown')
                if error_type not in recent_types:
                    recent_types[error_type] = 0
                recent_types[error_type] += 1
            
            most_recent = max(recent_types.items(), key=lambda x: x[1]) if recent_types else (None, 0)
            
            stats['recent_trend'] = {
                'most_common_type': most_recent[0],
                'count': most_recent[1],
                'percentage': (most_recent[1] / len(recent_errors)) * 100
            }
            
            # สถิติแยกตามประเภท
            for error_type, errors in error_types.items():
                stats['by_type'][error_type] = {
                    'count': len(errors),
                    'percentage': (len(errors) / len(self.error_history)) * 100,
                    'latest': errors[-1]['timestamp'] if errors else None
                }
        
        return stats

# สร้าง instance ของระบบเรียนรู้จากข้อผิดพลาด
error_learning_system = ErrorLearningSystem()

@app.route('/learning-stats', methods=['GET'])
def get_learning_stats():
    """ดึงสถิติการเรียนรู้ทั้งหมด"""
    try:
        stats = error_learning_system.get_error_statistics()
        rules = error_learning_system.get_improvement_rules()
        return jsonify({
            'error_statistics': stats,
            'improvement_rules': rules,
            'total_errors_processed': len(error_learning_system.error_history),
            'unique_error_types': list(error_learning_system.error_patterns.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/learning-suggestions', methods=['GET'])
def get_learning_suggestions():
    """ดึงคำแนะนำสำหรับ task_type ที่ระบุ"""
    task_type = request.args.get('task_type', '')
    try:
        suggestions = error_learning_system.suggest_improvements(task_type)
        return jsonify({
            'task_type': task_type,
            'suggestions': suggestions,
            'confidence_scores': error_learning_system.get_suggestion_confidence(task_type)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/error-history', methods=['GET'])
def get_error_history():
    """ดึงประวัติข้อผิดพลาดทั้งหมด"""
    try:
        history = error_learning_system.error_history
        return jsonify({
            'error_history': history,
            'total_entries': len(history),
            'summary': error_learning_system.summarize_error_history()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class Report:
    """คลาสสำหรับรายงานผลการตรวจสอบชุดข้อมูล"""
    def __init__(self, name="dataset", errors=None, warnings=None, stats=None):
        self.name = name
        self.errors = errors or []
        self.warnings = warnings or []
        self.stats = stats or {}
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # คำนวณคะแนนคุณภาพโดยรวม
        if len(self.errors) + len(self.warnings) == 0:
            self.overall_score = 100
        else:
            # คำนวณคะแนนโดยลดจาก 100 ตามจำนวนข้อผิดพลาด (errors มีผลมากกว่า warnings)
            error_penalty = min(len(self.errors) * 10, 80)
            warning_penalty = min(len(self.warnings) * 2, 20)
            self.overall_score = max(0, 100 - error_penalty - warning_penalty)
            
        # บันทึกจำนวนข้อผิดพลาดและคำเตือน
        self.error_count = len(self.errors)
        self.warning_count = len(self.warnings)
    
    def to_dict(self):
        """แปลงรายงานเป็น dict"""
        return {
            "dataset_name": self.name,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "statistics": self.stats
        }
    
    def to_json(self, filepath=None):
        """แปลงรายงานเป็น JSON"""
        report_dict = self.to_dict()
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            return filepath
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False)
    
    def to_csv(self, filepath=None):
        """แปลงรายงานเป็น CSV"""
        # แปลงข้อผิดพลาดและคำเตือนเป็น DataFrame
        errors_df = pd.DataFrame(self.errors) if self.errors else pd.DataFrame()
        warnings_df = pd.DataFrame(self.warnings) if self.warnings else pd.DataFrame()
        
        # เพิ่มคอลัมน์ประเภท
        if not errors_df.empty:
            errors_df['message_type'] = 'error'
        if not warnings_df.empty:
            warnings_df['message_type'] = 'warning'
        
        # รวม DataFrames
        combined_df = pd.concat([errors_df, warnings_df], ignore_index=True)
        
        # เพิ่มข้อมูลทั่วไป
        if not combined_df.empty:
            combined_df['dataset_name'] = self.name
            combined_df['timestamp'] = self.timestamp
            combined_df['overall_score'] = self.overall_score
        else:
            # สร้าง DataFrame ใหม่ถ้าไม่มีข้อผิดพลาดหรือคำเตือน
            combined_df = pd.DataFrame({
                'dataset_name': [self.name],
                'timestamp': [self.timestamp],
                'overall_score': [self.overall_score],
                'message': ['No errors or warnings found'],
                'message_type': ['info']
            })
        
        if filepath:
            combined_df.to_csv(filepath, index=False, encoding='utf-8')
            return filepath
        
        return combined_df.to_csv(index=False)
    
    def to_html(self, filepath=None):
        """แปลงรายงานเป็น HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dataset Quality Report: {self.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .score-badge {{ font-size: 24px; font-weight: bold; padding: 10px 15px; border-radius: 50%; display: inline-block; }}
                .good {{ background-color: #28a745; color: white; }}
                .average {{ background-color: #ffc107; color: black; }}
                .poor {{ background-color: #dc3545; color: white; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .error {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .stats-section {{ margin-top: 30px; }}
                .stats-item {{ margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>Dataset Quality Report</h1>
                <p>Dataset: <strong>{self.name}</strong></p>
                <p>Generated on: <strong>{self.timestamp}</strong></p>
                <p>Overall Quality Score: 
                    <span class="score-badge {
                    'good' if self.overall_score >= 80 else
                    'average' if self.overall_score >= 50 else
                    'poor'
                    }">{self.overall_score}</span>
                </p>
                <p>Errors: <strong class="error">{self.error_count}</strong> | Warnings: <strong class="warning">{self.warning_count}</strong></p>
            </div>
        """
        
        if self.errors:
            html_content += """
            <h2>Errors</h2>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Message</th>
                        <th>Field</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for error in self.errors:
                html_content += f"""
                <tr class="error">
                    <td>{error.get('type', 'Unknown')}</td>
                    <td>{error.get('message', '')}</td>
                    <td>{error.get('field', '')}</td>
                    <td>{error.get('details', '')}</td>
                </tr>
                """
                
            html_content += """
                </tbody>
            </table>
            """
            
        if self.warnings:
            html_content += """
            <h2>Warnings</h2>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Message</th>
                        <th>Field</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for warning in self.warnings:
                html_content += f"""
                <tr class="warning">
                    <td>{warning.get('type', 'Unknown')}</td>
                    <td>{warning.get('message', '')}</td>
                    <td>{warning.get('field', '')}</td>
                    <td>{warning.get('details', '')}</td>
                </tr>
                """
                
            html_content += """
                </tbody>
            </table>
            """
            
        if self.stats:
            html_content += """
            <div class="stats-section">
                <h2>Statistics</h2>
            """
            
            for key, value in self.stats.items():
                html_content += f"""
                <div class="stats-item">
                    <strong>{key}:</strong> {value}
                </div>
                """
                
            html_content += """
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return filepath
        
        return html_content
    
    def to_text(self, filepath=None):
        """แปลงรายงานเป็นข้อความธรรมดา"""
        text_content = f"""
DATASET QUALITY REPORT
======================
Dataset: {self.name}
Generated on: {self.timestamp}
Overall Quality Score: {self.overall_score}/100
Errors: {self.error_count}
Warnings: {self.warning_count}
"""
        
        if self.errors:
            text_content += """
ERRORS
------
"""
            for i, error in enumerate(self.errors, 1):
                text_content += f"""
{i}. Type: {error.get('type', 'Unknown')}
   Message: {error.get('message', '')}
   Field: {error.get('field', '')}
   Details: {error.get('details', '')}
"""
                
        if self.warnings:
            text_content += """
WARNINGS
--------
"""
            for i, warning in enumerate(self.warnings, 1):
                text_content += f"""
{i}. Type: {warning.get('type', 'Unknown')}
   Message: {warning.get('message', '')}
   Field: {warning.get('field', '')}
   Details: {warning.get('details', '')}
"""
                
        if self.stats:
            text_content += """
STATISTICS
----------
"""
            for key, value in self.stats.items():
                text_content += f"{key}: {value}\n"
                
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            return filepath
        
        return text_content
    
    def to_pdf(self, filepath=None):
        """แปลงรายงานเป็น PDF"""
        try:
            # ใช้ HTML ที่สร้างไว้แล้วเป็นต้นฉบับ
            html_content = self.to_html()
            
            # สร้างไฟล์ PDF
            from weasyprint import HTML
            
            if filepath:
                HTML(string=html_content).write_pdf(filepath)
                return filepath
            else:
                import io
                pdf_bytes = io.BytesIO()
                HTML(string=html_content).write_pdf(pdf_bytes)
                return pdf_bytes.getvalue()
                
        except ImportError:
            # ถ้าไม่มี weasyprint ให้แสดงข้อความแจ้งเตือน
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("PDF export requires weasyprint library. Please install it with 'pip install weasyprint'.")
                return filepath
            return "PDF export requires weasyprint library. Please install it with 'pip install weasyprint'."

@app.route('/preview-report/<path:filename>')
def preview_report(filename):
    """แสดงตัวอย่างรายงานแบบ HTML"""
    try:
        file_path = os.path.join(config.CHECK_RESULTS_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'ไม่พบไฟล์รายงาน'}), 404
            
        # ถ้าเป็นไฟล์ HTML สามารถแสดงโดยตรงได้
        if filename.endswith('.html'):
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        
        # ถ้าเป็นไฟล์ JSON ให้แปลงเป็น HTML
        elif filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # สร้าง Report object จากข้อมูล JSON
            report = Report(
                name=report_data.get('dataset_name', 'Dataset'),
                errors=report_data.get('errors', []),
                warnings=report_data.get('warnings', []),
                stats=report_data.get('statistics', {})
            )
            
            # แปลงเป็น HTML
            return report.to_html()
            
        # ถ้าเป็นไฟล์รูปแบบอื่น ให้แจ้งว่าไม่รองรับการแสดงตัวอย่าง
        else:
            return render_template('preview_error.html', 
                                  message=f'ไม่รองรับการแสดงตัวอย่างสำหรับไฟล์ประเภท: {os.path.splitext(filename)[1]}')
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาดในการแสดงตัวอย่างรายงาน: {str(e)}'}), 500

@app.route('/generate-from-source', methods=['POST'])
def generate_from_source():
    """สร้างข้อมูลจากไฟล์ต้นฉบับโดยไม่บิดเบือนข้อเท็จจริง"""
    try:
        # ตรวจสอบว่ามีไฟล์หรือไม่
        if not request.files:
            return jsonify({'error': 'ไม่มีไฟล์ที่อัปโหลด'}), 400
        
        # รับพารามิเตอร์
        task_type = request.form.get('task_type', '')
        prompt = request.form.get('prompt', '')
        output_format = request.form.get('output_format', 'json')
        strict_factual = request.form.get('strict_factual', 'true').lower() == 'true'
        quality_level = request.form.get('quality_level', 'standard')
        
        # ตรวจสอบรูปแบบไฟล์ที่รองรับ
        supported_formats = ['json', 'csv', 'parquet', 'arrow', 'txt']
        if output_format not in supported_formats:
            return jsonify({
                'error': f'รูปแบบไฟล์ไม่รองรับ รองรับเฉพาะ: {", ".join(supported_formats)}'
            }), 400
        
        # สร้างโฟลเดอร์ชั่วคราวสำหรับไฟล์ที่อัปโหลด
        temp_folder = os.path.join(config.UPLOAD_FOLDER, f"temp_{int(time.time())}")
        os.makedirs(temp_folder, exist_ok=True)
        
        # บันทึกไฟล์ทั้งหมด
        file_paths = []
        for key in request.files:
            file = request.files[key]
            if file.filename:
                file_path = os.path.join(temp_folder, secure_filename(file.filename))
                file.save(file_path)
                file_paths.append(file_path)
        
        if not file_paths:
            shutil.rmtree(temp_folder)  # ลบโฟลเดอร์ชั่วคราวถ้าไม่มีไฟล์
            return jsonify({'error': 'ไม่มีไฟล์ที่อัปโหลด'}), 400
        
        # อ่านข้อมูลจากไฟล์
        source_data = []
        for file_path in file_paths:
            try:
                # ตรวจสอบประเภทไฟล์และอ่านข้อมูล
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    source_data.append({'type': 'json', 'data': data, 'path': file_path})
                    
                elif file_ext == '.csv':
                    df = pd.read_csv(file_path)
                    source_data.append({'type': 'csv', 'data': df, 'path': file_path})
                    
                elif file_ext == '.parquet':
                    df = pd.read_parquet(file_path)
                    source_data.append({'type': 'parquet', 'data': df, 'path': file_path})
                    
                elif file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    source_data.append({'type': 'text', 'data': text, 'path': file_path})
                    
                elif file_ext in ['.pdf', '.docx', '.doc']:
                    # สำหรับไฟล์เอกสาร ใช้ document processor (ต้องติดตั้งไลบรารีเพิ่มเติม)
                    text = extract_text_from_document(file_path)
                    source_data.append({'type': 'document', 'data': text, 'path': file_path})
                    
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    # สำหรับไฟล์ภาพ ใช้ vision model วิเคราะห์
                    analysis = analyze_image_with_vision_model(file_path)
                    source_data.append({'type': 'image', 'data': analysis, 'path': file_path})
                    
                else:
                    # ไฟล์ประเภทอื่นๆ ให้แจ้งเตือนแต่ไม่หยุดการทำงาน
                    print(f"Warning: Unsupported file type: {file_ext} for {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        if not source_data:
            shutil.rmtree(temp_folder)  # ลบโฟลเดอร์ชั่วคราว
            return jsonify({'error': 'ไม่สามารถอ่านข้อมูลจากไฟล์ที่อัปโหลดได้'}), 400
        
        # สร้างคำสั่ง prompt สำหรับการ generate
        system_prompt = generate_factual_system_prompt(strict_factual, quality_level)
        source_content = prepare_source_content(source_data)
        generation_prompt = f"""
ข้อมูลต้นฉบับต่อไปนี้:

{source_content}

คำสั่ง: {prompt if prompt else 'กรุณาสร้างชุดข้อมูลที่มีคุณภาพจากข้อมูลต้นฉบับ โดยจัดระเบียบและเสริมเติมให้ข้อมูลสมบูรณ์'}

โปรดสร้างข้อมูลในรูปแบบ {task_type if task_type else 'ทั่วไป'} 
"""
        
        # สร้างข้อมูลใหม่จาก prompt
        api_key = os.environ.get('MISTRAL_API_KEY') or config.DEFAULT_API_KEY
        
        # ดึงคำแนะนำจากระบบเรียนรู้
        improvement_suggestions = error_learning_system.suggest_improvements('source_based_generation')
        
        # ปรับปรุง prompt ตามคำแนะนำ
        if improvement_suggestions:
            generation_prompt = f"{generation_prompt}\n\nคำแนะนำเพิ่มเติม:\n" + "\n".join(improvement_suggestions)
        
        # เรียกใช้ LLM
        try:
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_prompt}
            ]
            
            model_name = select_model_by_quality_level(quality_level)
            
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0.3,  # ค่าต่ำเพื่อให้ยึดตามข้อเท็จจริง
                max_tokens=4000
            )
            
            generated_text = response.choices[0].message.content
            
            # แปลงข้อความเป็นข้อมูลที่มีโครงสร้าง
            dataset = convert_to_structured_data(generated_text, output_format)
            
        except Exception as e:
            # ถ้าใช้ Mistral ไม่ได้ ให้ใช้ local model แทน
            print(f"Error using Mistral API: {str(e)}")
            dataset = fallback_generate_data_from_source(source_data, prompt, task_type, strict_factual)
        
        # ใช้กฎการปรับปรุงที่เรียนรู้มา
        dataset = error_learning_system.apply_improvements(dataset)
        
        # ตรวจสอบคุณภาพ
        checker = DatasetChecker(dataset, name="generated_from_source")
        report = checker.run_quality_check()
        
        # บันทึกข้อผิดพลาดที่พบ (ถ้ามี)
        if report.error_count > 0:
            for error in report.errors:
                error_learning_system.add_error({
                    'type': error.get('type', 'unknown'),
                    'field': error.get('field'),
                    'message': error.get('message'),
                    'task_type': 'source_based_generation',
                    'prompt': prompt
                })
        
        # สร้างชื่อไฟล์และบันทึก
        timestamp = int(time.time())
        filename = f"generated_from_source_{timestamp}.{output_format}"
        output_path = os.path.join(config.GENERATED_DATASET_DIR, filename)
        save_dataset(dataset, output_path, format=output_format)
        
        # บันทึกรายงาน
        report_filename = f"check_results_source_{timestamp}.html"
        report_path = os.path.join(config.CHECK_RESULTS_DIR, report_filename)
        report.to_html(report_path)
        
        # ลบโฟลเดอร์ชั่วคราว
        shutil.rmtree(temp_folder)
        
        # ดึงสถิติข้อผิดพลาด
        error_stats = error_learning_system.get_error_statistics()
        
        return jsonify({
            'message': 'สร้างข้อมูลจากไฟล์ต้นฉบับสำเร็จ',
            'source_files_count': len(file_paths),
            'quality_score': report.overall_score,
            'dataset_url': f"/download/{filename}",
            'report_url': f"/download/{report_filename}",
            'strict_factual': strict_factual,
            'quality_level': quality_level,
            'error_statistics': error_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_factual_system_prompt(strict_factual, quality_level):
    """สร้าง system prompt สำหรับการ generate ข้อมูลที่ยึดตามข้อเท็จจริง"""
    if strict_factual:
        return f"""คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญในการสร้างชุดข้อมูลคุณภาพสูงจากข้อมูลต้นฉบับ

กฎสำคัญ:
1. ยึดตามข้อเท็จจริงจากข้อมูลต้นฉบับเท่านั้น
2. ห้ามแต่งเติมข้อมูลที่ไม่มีในต้นฉบับ
3. ห้ามเปลี่ยนความหมายของข้อมูลต้นฉบับ
4. ห้ามแสดงความคิดเห็นหรืออคติส่วนตัว
5. คุณต้องจัดระเบียบข้อมูลให้อ่านง่ายและมีโครงสร้างที่ดี
6. ถ้าข้อมูลต้นฉบับไม่ครบถ้วน ให้ระบุว่าไม่มีข้อมูล แทนที่จะแต่งเติม

เน้นความถูกต้องและคุณภาพระดับ {quality_level}
"""
    else:
        return f"""คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญในการสร้างชุดข้อมูลคุณภาพสูงจากข้อมูลต้นฉบับ

กฎสำคัญ:
1. ยึดตามข้อเท็จจริงจากข้อมูลต้นฉบับเป็นหลัก
2. คุณสามารถเสริมแต่งรายละเอียดเพิ่มเติมได้ แต่ต้องไม่ขัดแย้งกับข้อมูลต้นฉบับ
3. ห้ามเปลี่ยนความหมายของข้อมูลต้นฉบับ
4. ห้ามแสดงความคิดเห็นหรืออคติส่วนตัว
5. คุณต้องจัดระเบียบข้อมูลให้อ่านง่ายและมีโครงสร้างที่ดี
6. ถ้าข้อมูลต้นฉบับไม่ครบถ้วน คุณสามารถเสริมเติมได้อย่างสมเหตุสมผล

เน้นความถูกต้องและคุณภาพระดับ {quality_level}
"""

def prepare_source_content(source_data):
    """เตรียมเนื้อหาจากข้อมูลต้นฉบับ"""
    content = ""
    
    for item in source_data:
        content += f"=== ไฟล์: {os.path.basename(item['path'])} (ประเภท: {item['type']}) ===\n\n"
        
        if item['type'] in ['json', 'csv', 'parquet']:
            if isinstance(item['data'], pd.DataFrame):
                # แปลง DataFrame เป็นข้อความ
                content += item['data'].to_string(index=False) + "\n\n"
            else:
                # แปลง JSON เป็นข้อความ
                content += json.dumps(item['data'], ensure_ascii=False, indent=2) + "\n\n"
        else:
            # สำหรับข้อความหรือข้อมูลอื่นๆ
            content += str(item['data']) + "\n\n"
    
    return content

def select_model_by_quality_level(quality_level):
    """เลือกโมเดลตามระดับคุณภาพ"""
    if quality_level == 'premium':
        return "mistral-large-latest"
    elif quality_level == 'standard':
        return "mistral-medium-latest"
    else:  # basic
        return "mistral-small-latest"

def convert_to_structured_data(text, output_format):
    """แปลงข้อความเป็นข้อมูลที่มีโครงสร้าง"""
    try:
        # พยายามแปลงเป็น JSON ก่อน
        # ค้นหาข้อมูล JSON ในข้อความ
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, text)
        
        if matches:
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    return data
                except:
                    continue
        
        # ถ้าไม่ใช่ JSON ที่ถูกต้อง ให้ลองวิธีอื่น
        # ถ้าข้อความมีรูปแบบคล้ายตาราง
        lines = text.strip().split('\n')
        if '|' in text:
            # อาจเป็นตาราง Markdown
            table_lines = [line for line in lines if line.strip().startswith('|')]
            if table_lines:
                # แปลงตาราง Markdown เป็น DataFrame
                header = table_lines[0].strip().split('|')[1:-1]
                header = [h.strip() for h in header]
                
                rows = []
                for line in table_lines[2:]:
                    cells = line.strip().split('|')[1:-1]
                    cells = [c.strip() for c in cells]
                    if len(cells) == len(header):
                        rows.append(cells)
                
                df = pd.DataFrame(rows, columns=header)
                
                # แปลงเป็นรูปแบบที่ต้องการ
                if output_format == 'json':
                    return json.loads(df.to_json(orient='records'))
                else:
                    return df
        
        # ถ้าไม่สามารถตีความโครงสร้างได้ ให้แปลงเป็นข้อความหรือ JSON อย่างง่าย
        if output_format == 'json':
            return {"content": text}
        else:
            df = pd.DataFrame({"content": [text]})
            return df
            
    except Exception as e:
        print(f"Error converting to structured data: {str(e)}")
        # ถ้าแปลงไม่ได้ ให้ส่งคืนเป็นข้อความ
        if output_format == 'json':
            return {"content": text}
        else:
            df = pd.DataFrame({"content": [text]})
            return df

def fallback_generate_data_from_source(source_data, prompt, task_type, strict_factual):
    """ฟังก์ชันสำรองในกรณีที่ใช้ API ไม่ได้"""
    # ในกรณีที่ใช้ API ไม่ได้ ให้สร้างข้อมูลอย่างง่ายจากข้อมูลต้นฉบับ
    result = {"source_files": []}
    
    for item in source_data:
        file_info = {
            "filename": os.path.basename(item['path']),
            "type": item['type']
        }
        
        if item['type'] in ['json', 'csv', 'parquet']:
            if isinstance(item['data'], pd.DataFrame):
                file_info["summary"] = f"DataFrame with {len(item['data'])} rows and {len(item['data'].columns)} columns"
                file_info["columns"] = item['data'].columns.tolist()
                file_info["sample"] = item['data'].head(5).to_dict(orient='records')
            else:
                file_info["summary"] = "JSON data"
                file_info["sample"] = item['data']
        else:
            # สำหรับข้อความหรือข้อมูลอื่นๆ
            content = str(item['data'])
            file_info["summary"] = f"Text content with {len(content)} characters"
            file_info["sample"] = content[:500] + "..." if len(content) > 500 else content
            
        result["source_files"].append(file_info)
    
    result["task_type"] = task_type
    result["prompt"] = prompt
    result["strict_factual"] = strict_factual
    result["generation_method"] = "fallback"
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    return result

def extract_text_from_document(file_path):
    """สกัดข้อความจากไฟล์เอกสาร"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # ใช้ PyPDF2 หรือ pdfplumber (ต้องติดตั้งเพิ่มเติม)
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                return f"ต้องการ PyPDF2 สำหรับอ่านไฟล์ PDF: {os.path.basename(file_path)}"
                
        elif file_ext in ['.docx', '.doc']:
            # ใช้ python-docx (ต้องติดตั้งเพิ่มเติม)
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                return f"ต้องการ python-docx สำหรับอ่านไฟล์ DOCX: {os.path.basename(file_path)}"
                
        else:
            return f"ไม่รองรับการอ่านไฟล์ประเภท: {file_ext}"
            
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True) 