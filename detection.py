import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# Load Hugging Face Open-Vocabulary Detector
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
).to(device)


def detect_target(
    image: Image.Image, target_text: str, confidence_threshold: float = 0.35
):
    """Returns True and the bounding box if target is found with sufficient confidence."""
    texts = [[f"a photo of a {target_text}"]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]

    if len(results["scores"]) > 0:
        # Highest confidence detection
        best_idx = torch.argmax(results["scores"])
        box = results["boxes"][best_idx].cpu().numpy()
        score = results["scores"][best_idx].cpu().item()
        return True, score, box

    return False, 0.0, None
