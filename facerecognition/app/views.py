from django.shortcuts import render
from app.forms import FaceRecognitionform
from app.machinelearning import pipeline_model
from django.conf import settings
from app.models import FaceRecognition
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import base64
import uuid
import numpy as np
import cv2

def index(request):   # ✅ OLD NAME (IMPORTANT)
    form = FaceRecognitionform()
    
    if request.method == 'POST':
        form = FaceRecognitionform(request.POST or None, request.FILES or None)
        
        if form.is_valid():
            save = form.save(commit=True)
            
            # get image path
            primary_key = save.pk
            imgobj = FaceRecognition.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)

            # run ML
            results = pipeline_model(filepath)
            print(results)

            return render(request, 'index.html', {
                'form': form,
                'upload': True,
                'results': results
            })
    
    return render(request, 'index.html', {
        'form': form,
        'upload': False
    })


@csrf_exempt
def webcam_capture(request):
    """Handle webcam image capture via AJAX POST."""
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image_data', '')
            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)

            # Remove data URL prefix (e.g. "data:image/png;base64,")
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            # Decode base64 to bytes
            img_bytes = base64.b64decode(image_data)

            # Convert to numpy array and then to OpenCV image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({'error': 'Could not decode image'}, status=400)

            # Save the webcam image to media/images/
            filename = 'webcam_{}.jpg'.format(uuid.uuid4().hex[:8])
            filepath = os.path.join(settings.MEDIA_ROOT, 'images', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, img)

            # Run ML pipeline
            results = pipeline_model(filepath)

            # Build response data
            response_data = {
                'success': True,
                'results': {
                    'count': results.get('count', []),
                    'face_detect_score': [float(s) for s in results.get('face_detect_score', [])],
                    'face_name': results.get('face_name', []),
                    'face_name_score': [float(s) for s in results.get('face_name_score', [])],
                    'emotion_name': results.get('emotion_name', []),
                    'emotion_name_score': [float(s) for s in results.get('emotion_name_score', [])],
                },
                'processed_image': '/media/ml_output/process.jpg',
                'roi_images': ['/media/ml_output/roi_{}.jpg'.format(c) for c in results.get('count', [])],
            }
            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST allowed'}, status=405)