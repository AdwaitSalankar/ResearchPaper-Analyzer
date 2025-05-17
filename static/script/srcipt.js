document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();
    const fileInput = document.getElementById("file-input").files[0];
    if (!fileInput) return;

    const formData = new FormData();
    formData.append("file", fileInput);

    fetch("/predict", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById("result");

            if (data.error) {
                resultDiv.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
            } else {
                let color = data.prediction === "Accepted" ? "green" : "red";
                resultDiv.innerHTML = `
                    <span style="color: ${color}; font-size: 20px;">
                        Prediction: ${data.prediction}
                    </span><br>
                    <span style="color: black; font-size: 18px;">
                        Score: ${data.score}% confidence
                    </span>
                `;
            }
        })
        .catch(error => console.error("Error:", error));
});