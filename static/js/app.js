const form = document.getElementById("restoreForm");
const input = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const previewWrap = document.getElementById("previewWrap");

const processButton = document.getElementById("processButton");

const progressSection = document.getElementById("progressSection");
const errorSection = document.getElementById("errorSection");
const resultsSection = document.getElementById("resultsSection");

const newImageButton = document.getElementById("newImageButton");
const progressTitle = document.getElementById("progressTitle");
const progressText = document.getElementById("progressText");

let progressTimer = null;


function showPreview(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        showError("Please select an image file.");
        return;
    }

    const reader = new FileReader();

    reader.onload = event => {
        previewWrap.innerHTML = `
            <img
                src="${event.target.result}"
                alt="Selected mural preview"
            >
        `;
    };

    reader.readAsDataURL(file);
}


input.addEventListener("change", () => {
    showPreview(input.files[0]);
});


["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });
});


["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
    });
});


dropZone.addEventListener("drop", event => {
    const files = event.dataTransfer.files;

    if (!files.length) return;

    input.files = files;

    showPreview(files[0]);
});


function showError(message) {
    errorSection.textContent = message;
    errorSection.classList.remove("hidden");
}


function hideError() {
    errorSection.classList.add("hidden");
    errorSection.textContent = "";
}


function setProcessing(isProcessing) {
    processButton.disabled = isProcessing;

    if (isProcessing) {
        processButton.textContent = "Processing…";
        progressSection.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        const startedAt = Date.now();
        progressText.textContent = (
            "Running U-Net++ and ZeroScratches detection, LaMa restoration and " +
            "super-resolution."
        );
        progressTimer = window.setInterval(() => {
            const elapsed = Math.floor((Date.now() - startedAt) / 1000);
            progressTitle.textContent = `Processing mural... ${elapsed}s`;
        }, 1000);
    } else {
        window.clearInterval(progressTimer);
        progressTimer = null;
        progressTitle.textContent = "Processing mural...";
        processButton.textContent = "Start AI Restoration";
        progressSection.classList.add("hidden");
    }
}


function cacheBust(url) {
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}t=${Date.now()}`;
}


form.addEventListener("submit", async event => {
    event.preventDefault();

    hideError();

    if (!input.files.length) {
        showError("Please choose a mural image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    setProcessing(true);

    try {
        const response = await fetch(
            "/api/process",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.error || "Processing failed."
            );
        }

        document.getElementById(
            "damagePercentage"
        ).textContent = `${data.damage_percentage}%`;

        document.getElementById(
            "thresholdValue"
        ).textContent = data.threshold;

        document.getElementById(
            "inputResolution"
        ).textContent = data.input_size;

        document.getElementById(
            "srResolution"
        ).textContent = data.sr_size;

        document.getElementById(
            "probMin"
        ).textContent = data.segmentation_probability.min;

        document.getElementById(
            "probMean"
        ).textContent = data.segmentation_probability.mean;

        document.getElementById(
            "probMax"
        ).textContent = data.segmentation_probability.max;


        const images = {
            originalResult: data.original,
            maskResult: data.mask,
            trainedMaskResult: data.trained_mask,
            scratchMaskResult: data.scratch_mask,
            overlayResult: data.overlay,
            restoredResult: data.restored,
            srResult: data.super_resolved
        };

        document.getElementById("trainedCoverage").textContent = `${data.detection.trained_percentage}%`;
        document.getElementById("scratchCoverage").textContent = `${data.detection.scratch_percentage}%`;
        document.getElementById("additionalCoverage").textContent = `${data.detection.additional_percentage}%`;
        document.getElementById("scratchThreshold").textContent = data.detection.scratch_threshold;
        document.getElementById("scratchProbMean").textContent = data.detection.scratch_probability.mean;
        document.getElementById("scratchProbMax").textContent = data.detection.scratch_probability.max;

        for (const [id, url] of Object.entries(images)) {
            document.getElementById(id).src = cacheBust(url);
        }

        const download = document.getElementById("downloadSR");

        download.href = data.super_resolved;


        resultsSection.classList.remove("hidden");

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        showError(error.message);
    } finally {
        setProcessing(false);
    }
});


newImageButton.addEventListener("click", () => {
    resultsSection.classList.add("hidden");

    input.value = "";

    previewWrap.innerHTML = `
        <div class="upload-icon">↑</div>
        <strong>Select mural image</strong>
        <span>or drag & drop it here</span>
    `;

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
});
