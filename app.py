import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
from datetime import datetime
import pickle
import uuid
import time
import shutil
from PIL import Image
import traceback
import sys

# Import utilities
from utils.background_removal import BackgroundRemover
from utils.edge_detection import EdgeDetector
from utils.preprocessing import ImagePreprocessor

# ============= DEFINE CUSTOM ACTIVATION FUNCTIONS =============
def mish_activation(x):
    """
    Mish activation function: x * tanh(softplus(x))
    """
    return x * tf.math.tanh(tf.math.softplus(x))

def swish_activation(x):
    """
    Swish activation function: x * sigmoid(x)
    """
    return x * tf.nn.sigmoid(x)

# Register custom activations
tf.keras.utils.get_custom_objects()['mish_activation'] = mish_activation
tf.keras.utils.get_custom_objects()['swish_activation'] = swish_activation
# ==============================================================

# Define custom metrics functions (same as during training)
def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    """R-squared (Coefficient of determination)"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return K.mean(K.abs(y_pred - y_true))

# Create Flask app
app = Flask(__name__)
app.secret_key = 'nitrosense-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_NAME'] = 'nitrosense_session'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize models and utilities
print("="*50)
print("Loading NitroSense AI Application")
print("="*50)

# ============= LOAD SOIL TEMPERATURE MEAN =============
print("📊 Loading soil temperature mean from training data...")
try:
    with open('utils/temperature_means.pkl', 'rb') as f:
        temp_means = pickle.load(f)
    SOIL_TEMP_MEAN = temp_means.get('soil_temp_mean', 29.8)
    print(f"✅ Using Soil Temperature mean: {SOIL_TEMP_MEAN}°C")
    print(f"ℹ️ Air Temperature will be provided by user input")
except Exception as e:
    print(f"⚠️ Could not load temperature means: {e}")
    print("   Using default soil temperature:")
    SOIL_TEMP_MEAN = 29.8
    print(f"   Soil Temp: {SOIL_TEMP_MEAN}°C")
# =====================================================

# Load U2Net model (portable version)
print("📦 Initializing Background Remover...")
try:
    background_remover = BackgroundRemover()
    print("✅ BackgroundRemover initialized (portable version)")
    print("   Weights should be at: u2net/weights/u2net.pth")
except Exception as e:
    print(f"⚠️ Error initializing BackgroundRemover: {e}")
    print("   Will use fallback mode (no background removal)")
    background_remover = None

# Load edge detector
print("📦 Initializing Edge Detector...")
try:
    edge_detector = EdgeDetector()
    print("✅ EdgeDetector initialized")
except Exception as e:
    print(f"⚠️ Error initializing EdgeDetector: {e}")
    edge_detector = None

# Load preprocessor - Use 224x224 to match DenseNet121 model
print("📦 Initializing Image Preprocessor...")
try:
    preprocessor = ImagePreprocessor(target_size=(224, 224))  # 224x224 for DenseNet121
    print("✅ ImagePreprocessor initialized with target_size=(224, 224)")
except Exception as e:
    print(f"⚠️ Error initializing ImagePreprocessor: {e}")
    preprocessor = None

# Load your trained model with custom metrics
print("📦 Loading Model...")
# Use the specific model file you have
model_path = os.path.join('models', 'Pyramid_fusion_densenet121_model.h5')

if os.path.exists(model_path):
    try:
        # Custom objects for loading
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'MSE': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'mae': mae,
            'MAE': mae,
            'mean_absolute_error': mae,
            'rmse': rmse,
            'RMSE': rmse,
            'root_mean_squared_error': rmse,
            'r2': r2,
            'R2': r2,
            'r_squared': r2,
            'R_squared': r2,
            # Add custom activation functions
            'mish_activation': mish_activation,
            'swish_activation': swish_activation,
        }
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        print(f"✅ Model loaded successfully from: {model_path}")
        
        # Verify model inputs
        print("📊 Model input structure:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i+1}: {input_layer.name} - Shape: {input_layer.shape}")
        
        # Recompile
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[mae, rmse, r2]
        )
        print("✅ Model recompiled with custom metrics")
        
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        traceback.print_exc()
        model = None
else:
    print(f"❌ Model file not found at: {model_path}")

# Load all scalers
print("📦 Loading All Scalers...")
try:
    # Load temperature scaler (for Avg_Temp and Soil)
    continuous_scaler_path = 'utils/scalers/continuous_scaler.pkl'
    if os.path.exists(continuous_scaler_path):
        continuous_scaler = pickle.load(open(continuous_scaler_path, 'rb'))
        print("✅ continuous_scaler loaded successfully")
    else:
        continuous_scaler = None
        print("⚠️ continuous_scaler not found")
    
    # Load days scaler
    days_scaler_path = 'utils/scalers/days_scaler.pkl'
    if os.path.exists(days_scaler_path):
        days_scaler = pickle.load(open(days_scaler_path, 'rb'))
        print("✅ days_scaler loaded successfully")
    else:
        days_scaler = None
        print("⚠️ days_scaler not found")
    
    # Load nitrogen mapping (used for display only)
    nitrogen_map_path = 'utils/scalers/nitrogen_map.pkl'
    if os.path.exists(nitrogen_map_path):
        nitrogen_map = pickle.load(open(nitrogen_map_path, 'rb'))
        print("✅ nitrogen_map loaded successfully")
    else:
        nitrogen_map = {0:0, 30:1, 60:2, 90:3, 120:4, 150:5, 180:6, 210:7}
        print("⚠️ nitrogen_map not found, using default mapping")
    
    # Load target scaler
    scaler_y_path = 'utils/scalers/scaler_y.pkl'
    if os.path.exists(scaler_y_path):
        scaler_y = pickle.load(open(scaler_y_path, 'rb'))
        print(f"✅ scaler_y loaded successfully")
        print(f"   Target scaler mean: {scaler_y.mean_[0]:.4f}")
        print(f"   Target scaler scale: {scaler_y.scale_[0]:.4f}")
    else:
        scaler_y = None
        print("⚠️ scaler_y not found")
    
except Exception as e:
    print(f"⚠️ Error loading scalers: {e}")
    continuous_scaler = None
    days_scaler = None
    nitrogen_map = {0:0, 30:1, 60:2, 90:3, 120:4, 150:5, 180:6, 210:7}
    scaler_y = None

print("" + "="*50)
print("✅ Application initialization complete!")
print("="*50)

def cleanup_old_images(max_age_hours=24):
    """Delete images older than max_age_hours"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            return
        current_time = time.time()
        for filename in os.listdir(upload_folder):
            filepath = os.path.join(upload_folder, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getctime(filepath)
                if file_age > max_age_hours * 3600:
                    os.remove(filepath)
                    print(f"🧹 Deleted old image: {filename}")
    except Exception as e:
        print(f"⚠️ Error cleaning up images: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_days(sowing_date, capture_date):
    """Calculate number of days between sowing and capture"""
    date_format = "%Y-%m-%d"
    sowing = datetime.strptime(sowing_date, date_format)
    capture = datetime.strptime(capture_date, date_format)
    days = (capture - sowing).days
    return max(days, 0)

def map_fertilizer_to_category(fertilizer_value):
    """
    Map continuous fertilizer value (0-210) to categorical bin (0-7)
    Based on the training preprocessing logic
    """
    nitrogen_bins = [0, 30, 60, 90, 120, 150, 180, 210]
    nitrogen_map = {0:0, 30:1, 60:2, 90:3, 120:4, 150:5, 180:6, 210:7}
    
    # Find closest bin
    closest_bin = min(nitrogen_bins, key=lambda x: abs(x - fertilizer_value))
    return nitrogen_map[closest_bin]

def classify_nitrogen_level(nitrogen_value):
    """
    Classify nitrogen content into categories
    nitrogen_value is in original scale (2-4.95%)
    """
    if nitrogen_value < 3.0:
        return {
            'category': 'Deficient',
            'message': '⚠️ Nitrogen Deficient - Fertilizer Recommended',
            'color': 'warning',
            'action': 'Apply nitrogen fertilizer'
        }
    elif nitrogen_value <= 4.0:
        return {
            'category': 'Sufficient',
            'message': '✅ Nitrogen Sufficient - No Fertilizer Needed',
            'color': 'success',
            'action': 'Maintain current practices'
        }
    else:
        return {
            'category': 'Excess',
            'message': '⚠️ Nitrogen Excess - Reduce Fertilizer Application',
            'color': 'danger',
            'action': 'Reduce or skip nitrogen application'
        }

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/estimation')
def estimation():
    """Estimation page route"""
    return render_template('estimation.html')
    
@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        print(f"📤 Upload - File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            # Ensure upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Generate filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save file
            file.save(filepath)
            print(f"💾 File saved to: {filepath}")
            
            # Verify file was saved
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                session['current_image'] = filename
                session.modified = True
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'image_url': f'/static/uploads/{filename}'
                })
            else:
                return jsonify({'error': 'Failed to save file'}), 400
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Clean up old images
    cleanup_old_images(24)
    
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        print(f"📨 Form data received: {data}")
        
        # ============= GET CONTINUOUS FERTILIZER VALUE =============
        try:
            fertilizer_amount = float(data.get('fertilizer'))
            # Validate range
            if fertilizer_amount < 0 or fertilizer_amount > 210:
                return jsonify({'error': 'Fertilizer amount must be between 0 and 210 kg/ha'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Please provide a valid fertilizer amount between 0-210 kg/ha'}), 400
        # ===========================================================
        
        # Get dates and calculate days
        sowing_date = data.get('sowingDate')
        capture_date = data.get('captureDate')
        days = calculate_days(sowing_date, capture_date)
        
        # Validate days range
        if days < 60 or days > 120:
            return jsonify({'error': f'Days must be between 60-120. Current: {days}'}), 400
        
        # Get air temperature from user input
        try:
            air_temp = float(data.get('airTemp'))
        except (TypeError, ValueError):
            return jsonify({'error': 'Please provide valid air temperature value'}), 400
        
        # Use soil temperature mean from training
        soil_temp = SOIL_TEMP_MEAN
        
        print(f"📝 Input parameters:")
        print(f"  Fertilizer: {fertilizer_amount} kg/ha")
        print(f"  Days after sowing: {days}")
        print(f"  Air Temperature: {air_temp}°C (user input)")
        print(f"  Soil Temperature: {soil_temp}°C (mean from training)")
        
        # ============= MAP FERTILIZER TO CATEGORICAL =============
        nitrogen_category = map_fertilizer_to_category(fertilizer_amount)
        print(f"  Fertilizer {fertilizer_amount} kg/ha → Category {nitrogen_category}")
        # ===========================================================
        
        # Check if all scalers are available
        if any(v is None for v in [continuous_scaler, days_scaler, scaler_y]):
            missing = []
            if continuous_scaler is None: missing.append("continuous_scaler")
            if days_scaler is None: missing.append("days_scaler")
            if scaler_y is None: missing.append("scaler_y")
            return jsonify({'error': f'Scalers not loaded: {", ".join(missing)}'}), 500
        
        # Get image from session
        filename = session.get('current_image')
        if not filename:
            return jsonify({'error': 'No image found. Please upload an image first.'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found'}), 400
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            # Try PIL as fallback
            try:
                pil_img = Image.open(image_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"✅ PIL fallback succeeded")
            except Exception as e:
                print(f"❌ All image reading methods failed: {e}")
                return jsonify({'error': 'Could not read image'}), 400
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Background removal - Use 224x224
        if background_remover is not None:
            try:
                bg_removed, mask = background_remover.remove_background(
                    image_rgb, 
                    target_size=(224, 224),  # 224x224 to match model
                    max_size=800
                )
                print(f"✅ Background removal completed")
            except Exception as e:
                print(f"⚠️ Background removal error: {e}")
                bg_removed = cv2.resize(image_rgb, (224, 224))
        else:
            bg_removed = cv2.resize(image_rgb, (224, 224))
        
        # Edge detection
        if edge_detector is not None:
            try:
                edge_image, edges = edge_detector.detect_edges(bg_removed)
            except Exception as e:
                print(f"⚠️ Edge detection error: {e}")
                edge_image = bg_removed
        else:
            edge_image = bg_removed
        
        # Final preprocessing - Use 224x224
        if preprocessor is not None:
            processed_image = preprocessor.preprocess_for_model(edge_image)
        else:
            processed_image = cv2.resize(edge_image, (224, 224)).astype(np.float32) / 255.0
        
        # Process tabular features
        # Create raw features
        temperature_features = np.array([[air_temp, soil_temp]], dtype=np.float32)
        days_array = np.array([[days]], dtype=np.float32)
        
        # Scale temperature features
        temperature_scaled = continuous_scaler.transform(temperature_features)
        
        # Map nitrogen level to categorical (0-7) using the mapping function
        nitrogen_categorical = np.array([[nitrogen_category]], dtype=np.float32)
        
        # Scale days
        days_scaled = days_scaler.transform(days_array)
        
        # Combine all features
        tabular_processed = np.concatenate([
            temperature_scaled,
            nitrogen_categorical,
            days_scaled
        ], axis=1).astype('float32')
        
        print(f"📊 Processed features shape: {tabular_processed.shape}")
        
        # ============= ENHANCED PREDICTION WITH DEBUGGING =============
        nitrogen_content = 3.62  # Default fallback
        prediction_successful = False

        print("" + "="*50)
        print("🔍 PREDICTION DEBUG INFORMATION")
        print("="*50)

        print(f"📊 Raw input values:")
        print(f"   - Air Temperature: {air_temp}°C")
        print(f"   - Soil Temperature: {soil_temp}°C (mean)")
        print(f"   - Fertilizer: {fertilizer_amount} kg/ha → Category {nitrogen_category}")
        print(f"   - Days: {days}")

        if continuous_scaler is not None:
            print(f"📊 Temperature scaler stats:")
            print(f"   - Mean: {continuous_scaler.mean_}")
            print(f"   - Scale: {continuous_scaler.scale_}")

        if days_scaler is not None:
            print(f"📊 Days scaler stats:")
            print(f"   - Min: {days_scaler.data_min_[0]:.2f}")
            print(f"   - Max: {days_scaler.data_max_[0]:.2f}")

        if model is not None:
            try:
                # Image stats
                print(f"📊 Image stats:")
                print(f"   - Shape: {processed_image.shape}")
                print(f"   - Min: {processed_image.min():.3f}")
                print(f"   - Max: {processed_image.max():.3f}")
                print(f"   - Mean: {processed_image.mean():.3f}")
                
                image_input = np.expand_dims(processed_image, axis=0)
                feature_input = np.expand_dims(tabular_processed[0], axis=0)
                
                print(f"🔄 Model input shapes:")
                print(f"   - Image input: {image_input.shape}")
                print(f"   - Feature input: {feature_input.shape}")
                print(f"   - Feature values: {feature_input[0]}")
                
                # Make prediction
                prediction = model.predict([image_input, feature_input], verbose=0)
                raw_prediction = prediction[0][0]
                print(f"🔍 Raw model output (normalized): {raw_prediction:.6f}")
                
                # Denormalize using scaler_y
                if scaler_y is not None:
                    print(f"📊 Target scaler stats:")
                    print(f"   - Mean: {scaler_y.mean_[0]:.4f}")
                    print(f"   - Scale: {scaler_y.scale_[0]:.4f}")
                    
                    nitrogen_content = scaler_y.inverse_transform(prediction)[0][0]
                    print(f"🔍 After inverse_transform: {nitrogen_content:.4f}%")
                    prediction_successful = True
                    
                    # Validate range
                    if nitrogen_content < 1.0 or nitrogen_content > 6.0:
                        print(f"⚠️ Warning: Predicted nitrogen {nitrogen_content:.2f}% is outside expected range")
                        print(f"   Expected range: 2.0 - 4.95%")
                        nitrogen_content = 3.62
                        prediction_successful = False
                else:
                    nitrogen_content = raw_prediction
                    prediction_successful = True
                    
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                traceback.print_exc()
                nitrogen_content = 3.62
        else:
            print("⚠️ Model not loaded, using default value")
            nitrogen_content = 3.62

        print("="*50)
        print(f"✅ Final nitrogen content: {nitrogen_content:.2f}%")
        # ===========================================================================
        
        # Classify
        classification = classify_nitrogen_level(nitrogen_content)
        
        # Store in session and redirect
        session['prediction_result'] = {
            'nitrogen_content': round(float(nitrogen_content), 2),
            'fertilizer_applied': fertilizer_amount,
            'days': days,
            'classification': classification,
            'prediction_successful': prediction_successful
        }
        session.modified = True
        
        return redirect(url_for('result'))
    
    except Exception as e:
        print(f"❌ Prediction route error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    """Result page route"""
    prediction_result = session.get('prediction_result', None)
    
    if prediction_result is None:
        return redirect(url_for('estimation'))
    
    return render_template('result.html', result=prediction_result)
    
# ==================== NAVIGATION ROUTES ====================
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('team.html')
# ==================== END OF ROUTES ====================

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"📁 Upload folder ready: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 U2Net weights should be at: u2net/weights/u2net.pth")
    print(f"🚀 Starting NitroSense AI application...")
    print(f"   Access at: http://localhost:5000")
    print(f"   Press CTRL+C to quit")
    
    # Run the app with proper settings
    try:
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False  # Disable reloader to avoid watchdog issues
        )
    except SystemExit:
        print("👋 Application stopped normally")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        traceback.print_exc()
