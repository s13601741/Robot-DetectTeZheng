import cv2 
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
vid = cv2.VideoCapture(0) 

while(True): 
    ret, frame = vid.read() 
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    
    if key == ord('s'):
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        pil_image = Image.fromarray(color_coverted)
        text = input("Question:")
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        # prepare inputs
        encoding = processor(pil_image, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", model.config.id2label[idx])
        
vid.release() 
cv2.destroyAllWindows() 

