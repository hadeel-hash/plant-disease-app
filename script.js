const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

const predictButton = document.getElementById("predictButton");
const result = document.getElementById("result");


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file && file.type.startsWith("image/")) {

        preview.src = URL.createObjectURL(file);

        predictButton.disabled = false;

        result.textContent = "Image is ready for prediction.";

    } else {

        preview.src = "";

        predictButton.disabled = true;

        result.textContent = "Please select a valid image.";
    }

});


predictButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        result.textContent = "Please select a plant image.";

        return;
    }


    const formData = new FormData();

    formData.append("file", file);


    result.textContent = "Analyzing image...";

    predictButton.disabled = true;


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            throw new Error("Prediction request failed.");
        }


        const data = await response.json();


        const confidence =
            (data.confidence * 100).toFixed(2);


        result.textContent =
            `Prediction: ${data.disease}
Confidence: ${confidence}%`;


    } catch (error) {

        result.textContent =
            "Could not connect to the server.";

        console.error(error);

    } finally {

        predictButton.disabled = false;
    }

});