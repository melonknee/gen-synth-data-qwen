# Fixed Qwen 2B Instruct Image Captioning for Flickr30K Dataset
# This version properly handles image inputs to generate unique captions

# Step 1: Install required packages
"""
Run these commands first:
pip install transformers torch torchvision pillow datasets accelerate
pip install qwen-vl-utils  # For proper image processing
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # This is crucial for image processing
from PIL import Image
import os
import json
from tqdm import tqdm
import requests

# Step 2: Setup and load the Qwen-VL model (CORRECTED VERSION)
print("Loading Qwen2-VL-2B-Instruct model...")

model_name = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the correct model class for vision-language tasks
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
    device_map="auto",
    trust_remote_code=True
)

# Load processor (handles both text and images)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print(f"Model loaded on: {next(model.parameters()).device}")

# Step 3: Correct image processing and caption generation
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

# Step 4: Test function to verify the model works with different images
def test_model_with_samples():
    """
    Test the model with a few images to make sure it generates different captions
    """
    print("Testing model with sample images...")
    
    # You can test with URLs first
    test_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    ]
    
    for i, url in enumerate(test_urls):
        try:
            # Download image
            response = requests.get(url)
            temp_path = f"temp_test_{i}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Generate caption
            caption = generate_caption_with_image(temp_path, model, processor)
            print(f"Test Image {i+1}: {caption}")
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            print(f"Test {i+1} failed: {e}")
    
    print("-" * 50)

# Step 5: Improved batch processing
def process_flickr30k_images_fixed(image_paths, output_file="qwen_captions_fixed.json", batch_size=1):
    """
    Process images with proper error handling and verification
    """
    results = []
    successful = 0
    failed = 0
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Generating captions")):
        try:
            # Verify image exists and is readable
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                failed += 1
                continue
            
            # Generate caption
            caption = generate_caption_with_image(img_path, model, processor)
            
            # Verify we got a meaningful caption (not an error)
            if caption.startswith("Error:"):
                print(f"Failed to process {img_path}: {caption}")
                failed += 1
                continue
            
            # Store result
            result = {
                "image_path": img_path,
                "image_id": os.path.basename(img_path),
                "generated_caption": caption,
                "caption_length": len(caption.split())
            }
            results.append(result)
            successful += 1
            
            # Print progress every 5 images
            if (i + 1) % 5 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images. Success: {successful}, Failed: {failed}")
                print(f"Latest caption: {caption[:100]}...")
                
                # Save intermediate results
                with open(f"temp_{output_file}", 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Unexpected error with {img_path}: {e}")
            failed += 1
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Success: {successful}, Failed: {failed}")
    print(f"Results saved to {output_file}")
    
    return results

# Step 6: Verification function
def verify_caption_diversity(results_file):
    """
    Check if the generated captions are actually different
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    captions = [item['generated_caption'] for item in results]
    unique_captions = set(captions)
    
    print(f"Total captions: {len(captions)}")
    print(f"Unique captions: {len(unique_captions)}")
    print(f"Diversity ratio: {len(unique_captions)/len(captions):.2%}")
    
    # Show some sample captions
    print("\nSample captions:")
    for i, item in enumerate(results[:5]):
        print(f"{i+1}. {item['image_id']}: {item['generated_caption'][:100]}...")
    
    if len(unique_captions) < len(captions) * 0.8:
        print("\n⚠️  WARNING: Low caption diversity detected!")
        print("This might indicate the model isn't processing images properly.")
    else:
        print("\n✅ Good caption diversity - model appears to be working correctly!")

# Step 7: Main execution with debugging
def main():
    """
    Main function with proper testing and debugging
    """
    print("=== Fixed Qwen2-VL Image Captioning ===")
    
    # Step 1: Test the model first
    test_model_with_samples()
    
    # Step 2: Process your actual images
    IMAGE_DIR = "data/flickr30k_images"  # Update this path
    OUTPUT_FILE = "qwen_captions_fixed.json"
    
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        print("Please update the IMAGE_DIR path to your Flickr30K images folder")
        return
    
    # Get image paths
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(IMAGE_DIR, filename))
    
    if not image_paths:
        print(f"❌ No images found in {IMAGE_DIR}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Start with a small batch for testing
    test_batch = image_paths[:10]  # Test with first 10 images
    print(f"Processing test batch of {len(test_batch)} images...")
    
    results = process_flickr30k_images_fixed(test_batch, "test_" + OUTPUT_FILE)
    
    # Verify the results
    if results:
        verify_caption_diversity("test_" + OUTPUT_FILE)
        
        # If test batch looks good, ask user if they want to continue
        response = input(f"\nTest batch completed successfully! Process all {len(image_paths)} images? (y/n): ")
        if response.lower() == 'y':
            print("Processing full dataset...")
            full_results = process_flickr30k_images_fixed(image_paths, OUTPUT_FILE)
            verify_caption_diversity(OUTPUT_FILE)
    else:
        print("❌ Test batch failed. Please check the error messages above.")

# Additional debugging function
def debug_single_image(image_path):
    """
    Debug a single image to see exactly what's happening
    """
    print(f"Debugging image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print("❌ File does not exist")
        return
    
    # Try to load image
    try:
        img = Image.open(image_path)
        print(f"✅ Image loaded successfully: {img.size}, {img.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Try to generate caption
    try:
        caption = generate_caption_with_image(image_path, model, processor)
        print(f"✅ Caption generated: {caption}")
    except Exception as e:
        print(f"❌ Error generating caption: {e}")

if __name__ == "__main__":
    main()

# Quick fix test - uncomment this to debug a specific image
# debug_single_image("path/to/your/test/image.jpg")