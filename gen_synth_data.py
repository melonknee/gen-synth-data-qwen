import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # This is crucial for image processing
from PIL import Image
import os
from datasets import load_dataset
import json
from tqdm import tqdm

#################### LOADING MODEL
print("Loading Qwen 2B Instruct model...")
# Load the model and tokenizer
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Vision-Language version
device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto",  # Automatically handle device placement
    trust_remote_code=True
)

# Load processor (handles both text and images)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print(f"Model loaded on: {next(model.parameters()).device}")
print(f"Model loaded on: {device}")



#################### LOADING Flickr30k DATASET
print("Loading Flickr30K dataset...")

# If you wanna use hf datasets (if available)
def load_hf_flickr30k():
    """Load Flickr30K from HuggingFace datasets"""
    try:
        dataset = load_dataset("nlphuji/flickr30k", split="test[:100]")  # Load first 100 for testing
        return dataset
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        return None

# OR if you have local Flickr30k files
def load_local_flickr30k(image_dir, annotations_file):
    """
    Load local Flickr30K dataset
    image_dir: path to folder containing images
    annotations_file: path to captions/annotations file
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    image_paths = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(image_dir, img_file))
    
    return image_paths, annotations


#################### IMAGE PREPROCESSING
def preprocess_image(image_path):
    """
    Load and preprocess image for the model
    """
    try:
        image = Image.open(image_path).convert("RGB")
        # Qwen models typically handle resizing internally
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    

##### CAPTION GENERATION FUNCTION

def generate_caption_with_image(image_path, model, processor, max_length=100):
    """
    Generate caption for a single image using Qwen2-VL model
    This version properly processes the image!
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Create the proper message format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # Pass the PIL image directly
                    },
                    {
                        "type": "text", 
                        "text": "Describe this image in detail. Focus on what you can see - people, objects, actions, setting, and mood."
                    },
                ],
            }
        ]
        
        # Process the input properly
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process both image and text
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate the caption
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.1,  # Reduce repetition
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Trim the input tokens to get only the generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated caption
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return caption.strip()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"Error: {str(e)}"

    


    ### BATCH PROCESSING FUNCTION
def process_flickr30k_images(image_paths, output_file="generated_captions.json"):
    """
    Process all images and generate captions
    """
    results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, img_path in enumerate(tqdm(image_paths)):
        # Load image
        image = preprocess_image(img_path)
        if image is None:
            continue
            
        # Generate caption
        caption = generate_caption_with_image(image, model, processor, tokenizer)
        
        # Store result
        result = {
            "image_path": img_path,
            "image_id": os.path.basename(img_path),
            "generated_caption": caption
        }
        results.append(result)
        
        # Save progress every 10 images
        if (i + 1) % 10 == 0:
            with open(f"temp_{output_file}", 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Captions saved to {output_file}")
    return results


### EXAMPLE EXECUTION
def main():
    """
    Main function to run the captioning process
    """
    
    # Configure your paths here
    IMAGE_DIR = "data\\flickr30k_images"  # Replace with your image directory
    ANNOTATIONS_FILE = "data\flickr30k_images"  # Replace with annotations file
    OUTPUT_FILE = "qwen_generated_captions.json"
    
    print("=== Qwen 2B Instruct Image Captioning for Flickr30K ===")
    
    # Method 1: Using local files
    if os.path.exists(IMAGE_DIR):
        print("Using local Flickr30K files...")
        image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit to first 50 for testing
        image_paths = image_paths[:50]
        
        # Process images
        results = process_flickr30k_images(image_paths, OUTPUT_FILE)
        
    else:
        print("Local directory not found. Using sample images...")
        # For demonstration, create a sample processing
        sample_images = ["sample1.jpg", "sample2.jpg"]  # Replace with actual paths
        
        for img_path in sample_images:
            if os.path.exists(img_path):
                image = preprocess_image(img_path)
                caption = generate_caption(image, model, tokenizer)
                print(f"Image: {img_path}")
                print(f"Caption: {caption}")
                print("-" * 50)





# Step 8: Evaluation helpers (optional)
def evaluate_captions(generated_file, reference_file=None):
    """
    Optional: Evaluate generated captions against reference captions
    """
    with open(generated_file, 'r') as f:
        generated = json.load(f)
    
    print(f"Generated {len(generated)} captions")
    
    # Show some examples
    print("\nSample generated captions:")
    for i, item in enumerate(generated[:5]):
        print(f"{i+1}. Image: {item['image_id']}")
        print(f"   Caption: {item['generated_caption']}")
        print()

if __name__ == "__main__":
    main()