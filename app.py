import os
import uuid
import traceback

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for
)
from werkzeug.utils import secure_filename

from utils.pipeline import MuralRestorationPipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


SEGMENTATION_MODEL = os.path.join(
    MODEL_DIR,
    "mural_damage_unetplusplus_efficientnetb3.pth"
)

SUPER_RESOLUTION_MODEL = os.path.join(
    MODEL_DIR,
    "final_mural_edsr_x4_dynamic.keras"
)

SEGMENTATION_METADATA = os.path.join(
    MODEL_DIR,
    "segmentation_metadata.json"
)

RESTORATION_MODEL = os.path.join(MODEL_DIR, "big-lama.pt")
SCRATCH_MODEL_DIR = os.path.join(MODEL_DIR, "zeroscratches")


pipeline = None
pipeline_error = None

try:
    pipeline = MuralRestorationPipeline(
        segmentation_model_path=SEGMENTATION_MODEL,
        super_resolution_model_path=SUPER_RESOLUTION_MODEL,
        segmentation_metadata_path=SEGMENTATION_METADATA,
        restoration_model_path=RESTORATION_MODEL,
        scratch_model_dir=SCRATCH_MODEL_DIR
    )
except Exception as exc:
    pipeline_error = str(exc)
    print("Pipeline initialization failed:")
    traceback.print_exc()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template(
        "index.html",
        pipeline_ready=(pipeline is not None),
        pipeline_error=pipeline_error
    )


@app.route("/api/status")
def status():
    return jsonify({
        "ready": pipeline is not None,
        "error": pipeline_error
    })


@app.route("/api/process", methods=["POST"])
def process_image():

    if pipeline is None:
        return jsonify({
            "success": False,
            "error": (
                "Models are not loaded. Check the segmentation and EDSR models, "
                "run python scripts/download_lama.py and "
                "python scripts/download_zeroscratches.py, and restart Flask."
            )
        }), 500

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image was uploaded."
        }), 400

    uploaded = request.files["image"]

    if uploaded.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    if not allowed_file(uploaded.filename):
        return jsonify({
            "success": False,
            "error": "Supported formats: PNG, JPG, JPEG, BMP and WEBP."
        }), 400

    original_name = secure_filename(uploaded.filename)
    extension = original_name.rsplit(".", 1)[1].lower()

    run_id = uuid.uuid4().hex[:12]

    stored_name = f"{run_id}_original.{extension}"
    upload_path = os.path.join(UPLOAD_DIR, stored_name)

    uploaded.save(upload_path)

    try:
        result = pipeline.process(
            image_path=upload_path,
            output_dir=RESULT_DIR,
            run_id=run_id
        )

        response = {
            "success": True,
            "run_id": run_id,
            "restoration_method": result["restoration_method"],
            "detection": result["detection"],
            "trained_mask": url_for(
                "static",
                filename=f"results/{os.path.basename(result['trained_mask_path'])}"
            ),
            "scratch_mask": url_for(
                "static",
                filename=f"results/{os.path.basename(result['scratch_mask_path'])}"
            ),
            "damage_percentage": result["damage_percentage"],
            "threshold": result["threshold"],
            "segmentation_probability": result["segmentation_probability"],
            "original": url_for(
                "static",
                filename=f"uploads/{stored_name}"
            ),
            "mask": url_for(
                "static",
                filename=f"results/{os.path.basename(result['mask_path'])}"
            ),
            "overlay": url_for(
                "static",
                filename=f"results/{os.path.basename(result['overlay_path'])}"
            ),
            "restored": url_for(
                "static",
                filename=f"results/{os.path.basename(result['restored_path'])}"
            ),
            "super_resolved": url_for(
                "static",
                filename=f"results/{os.path.basename(result['sr_path'])}"
            ),
            "input_size": result["input_size"],
            "sr_size": result["sr_size"],
            "timings": result["timings"]
        }

        print(
            f"Run {run_id} completed in {result['timings']['total']}s "
            f"(segmentation {result['timings']['segmentation']}s, "
            f"restoration {result['timings']['restoration']}s, "
            f"super-resolution "
            f"{result['timings']['super_resolution']}s)"
        )

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "success": False,
        "error": "Image is larger than 20 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        use_reloader=False
    )
