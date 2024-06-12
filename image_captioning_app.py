import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_images_from_url(url, image=None):
  """
  This function accepts both URL (optional) and image (optional) arguments.
  """
  captions = []
  if url:
      # Process webpage images if a URL is provided
      response = requests.get(url)
      soup = BeautifulSoup(response.text, 'html.parser')
      img_elements = soup.find_all('img')
      for img_element in img_elements:
          img_url = img_element.get('src')
          # ... (rest of logic for processing images from URL) ...
          if 'svg' in img_url or '1x1' in img_url:
              continue
          elif '//' in img_url:
              img_url = 'https:' + img_url
          elif not img_url.startswith('http://') and not img_url.startswith('https://'):
              continue
          try:
              response = requests.get(img_url)
              raw_image = Image.open(BytesIO(response.content))
              if raw_image.size[0] * raw_image.size[1] < 400:
                  continue
              raw_image = raw_image.convert('RGB')
              inputs = processor(images=raw_image, return_tensors="pt")
              out = model.generate(**inputs, max_new_tokens=50)
              caption = processor.decode(out[0], skip_special_tokens=True)
              captions.append((img_url, caption))
          except Exception as e:
              print(f"Error processing image {img_url}: {e}")
              continue
  elif image:
      # Process uploaded image if available
      if not isinstance(image, Image.Image):
          image = Image.open(BytesIO(image))
      raw_image = image.convert('RGB')
      inputs = processor(images=raw_image, return_tensors="pt")
      out = model.generate(**inputs, max_new_tokens=50)
      caption = processor.decode(out[0], skip_special_tokens=True)
      captions.append((None, caption))  # No URL for uploaded image
  return captions

def display_captions(url, image=None):
  captions = caption_images_from_url(url, image)
  html_output = ""
  for img_url, caption in captions:
      html_output += f"<p>"
      if img_url:
          html_output += f"<img src='{img_url}' width='200' />"
      html_output += f"<br>{caption}</p>"
  return html_output

# Define the Gradio interface
interface = gr.Interface(
  fn=display_captions,
  inputs=[
      gr.Textbox(lines=1, placeholder="Enter a URL to a webpage (optional)"),
      gr.Image(type="pil", label="Upload an Image (optional)"),
  ],
  outputs=gr.HTML(),
  title="Image Captioning",
  description="Upload an image or enter a webpage URL (optional) to generate captions."
)

# Launch the app with share=True to create a public link
if __name__ == "__main__":
  interface.launch(share=True)
