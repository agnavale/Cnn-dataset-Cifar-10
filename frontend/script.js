function uploadImage() {
            let fileInput = document.getElementById("fileInput").files[0];
            let preview = document.getElementById("preview");
            let result = document.getElementById("result");
            let error = document.getElementById("error");
            let loading = document.getElementById("loading");

            if (!fileInput) {
                error.innerText = "Please select an image.";
                return;
            }

            let formData = new FormData();
            formData.append("file", fileInput);

            // Clear previous results
            result.innerText = "";
            error.innerText = "";
            loading.style.display = "block";

            fetch("http://127.0.0.1:8000/predict/", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = "none";
                result.innerText = `Prediction: ${data.prediction}`;
            })
            .catch(err => {
                loading.style.display = "none";
                error.innerText = "Error: Unable to get prediction.";
                console.error("Error:", err);
            });

            // Show image preview
            let reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(fileInput);
        }
