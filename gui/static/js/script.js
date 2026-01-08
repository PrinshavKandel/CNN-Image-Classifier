// --- Global Elements ---
const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');
const previewImage = document.getElementById('preview-image');
const loadingOverlay = document.getElementById('loading');

// Result display elements
const predictionElement = document.getElementById('prediction');
const accuracyElement = document.getElementById('accuracy');
const precisionElement = document.getElementById('precision');
const recallElement = document.getElementById('recall');
const f1ScoreElement = document.getElementById('f1-score');

// Chart and Gauge elements
const resultsChartCanvas = document.getElementById('results-chart');
const gaugeCanvas = document.getElementById('gauge-canvas');

let currentFileName = null;
let currentChart = null;
let currentGauge = null;



//  FILE INPUT AND PREVIEW 

fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();

        reader.onload = function(e) {
            previewImage.src = e.target.result;
            // Clear any previous results when a new image is loaded
            resetResultsDisplay();
        };
        
        reader.readAsDataURL(file);
        
        // Enable the prediction button once a file is selected
        predictBtn.disabled = false;
    }
});


//  PREDICTION BUTTON 

predictBtn.addEventListener('click', async () => {
    if (!fileInput.files[0]) {
        alert("Please select an image file first.");
        return;
    }

    // Disable button and  loading screen
    predictBtn.disabled = true;
    showLoading(true);

    try {
        // --- Step A: Upload the file to the server ---
        const uploadedFilename = await uploadFile(fileInput.files[0]);
        
        if (uploadedFilename) {
            currentFileName = uploadedFilename;

            // --- Step B: Request prediction using the uploaded filename ---
            const predictionResults = await getPrediction(currentFileName);
            
            // --- Step C: Update the UI with results ---
            updateResultsUI(predictionResults);
        }

    } catch (error) {
        console.error("Prediction process failed:", error);
        alert(`Error during prediction: ${error.message}`);
    } finally {
        // Re-enable button and hide loading screen
        predictBtn.disabled = false;
        showLoading(false);
    }
});



// 3. API CALL FUNCTIONS


/** Sends the file to the /upload route and returns the server-saved filename. */
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    if (response.ok && data.success) {
        return data.filename;
    } else {
        throw new Error(data.error || 'Failed to upload image.');
    }
}

/** Sends the filename to the /predict route and returns the prediction data. */
async function getPrediction(filename) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: filename })
    });

    const data = await response.json();
    if (response.ok && !data.error) {
        return data;
    } else {
        throw new Error(data.error || 'Prediction failed on the server.');
    }
}



// 4. UI UPDATE FUNCTIONS


function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function resetResultsDisplay() {
    predictionElement.textContent = '...';
    predictionElement.className = 'metric-value'; // Reset color class
    accuracyElement.textContent = '';
    precisionElement.textContent = '';
    recallElement.textContent = '';
    f1ScoreElement.textContent = '';
    
    if (currentChart) currentChart.destroy();
    if (currentGauge) currentGauge.destroy();
}


/** Updates all metric display elements and initializes charts. */
function updateResultsUI(results) {
    // A. Update Metrics
    predictionElement.textContent = results.prediction;
    accuracyElement.textContent = results.accuracy;
    precisionElement.textContent = results.precision;
    recallElement.textContent = results.recall;
    f1ScoreElement.textContent = results.f1_score;

    // Set color based on prediction (using classes defined in style.css)
    const isPositive = results.prediction.toLowerCase().includes('positive');
    predictionElement.classList.add(isPositive ? 'positive' : 'negative');
    
    // B. Update Charts
    initResultsChart(results.graph_data.labels, results.graph_data.values);
    initGauge(results.probability);
}


// 5. CHART.JS INITIALIZATION


/** Initializes the bar chart showing class probabilities. */
function initResultsChart(labels, values) {
    if (currentChart) currentChart.destroy();

    const chartData = {
        labels: labels,
        datasets: [{
            label: 'Confidence Score',
            data: values,
            backgroundColor: values.map((v) => v === Math.max(...values) ? 'rgba(0, 123, 255, 0.8)' : 'rgba(108, 117, 125, 0.6)'),
            borderColor: values.map((v) => v === Math.max(...values) ? '#007bff' : '#6c757d'),
            borderWidth: 1
        }]
    };

    currentChart = new Chart(resultsChartCanvas, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: { display: true, text: 'Probability' }
                }
            }
        }
    });
}

/** Initializes a simple gauge chart (Doughnut Chart variation). */
function initGauge(probability) {
    if (currentGauge) currentGauge.destroy();

    const value = probability;
    const data = {
        datasets: [{
            data: [value, 1 - value],
            backgroundColor: [
                value > 0.7 ? '#dc3545' : (value > 0.4 ? '#ffc107' : '#28a745'),
                '#343a40' // Background color for the remaining arc
            ],
            borderWidth: 0
        }]
    };

    currentGauge = new Chart(gaugeCanvas, {
        type: 'doughnut',
        data: data,
        options: {
            rotation: 270, // Start from the bottom left
            circumference: 180, // Half circle
            responsive: true,
            maintainAspectRatio: true,
            cutout: '80%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false },
                // Custom plugin to display the value in the center
                datalabels: {
                    formatter: (value, context) => {
                        // Only show label for the probability segment
                        if (context.dataIndex === 0) {
                            return `${(probability * 100).toFixed(1)}%`;
                        }
                        return '';
                    },
                    color: '#e0e0e0',
                    font: { size: 24, weight: 'bold' },
                    anchor: 'center',
                    align: 'center'
                }
            }
        },
        plugins: [{
            id: 'textCenter',
            beforeDraw: function(chart) {
                const width = chart.width,
                      height = chart.height,
                      ctx = chart.ctx;
                
                ctx.restore();
                const fontSize = (height / 150).toFixed(2);
                ctx.font = fontSize + "em sans-serif";
                ctx.textBaseline = "middle";
                
                const text = `${(probability * 100).toFixed(1)}%`;
                const textX = Math.round((width - ctx.measureText(text).width) / 2);
                // Adjust position for gauges
                const textY = (height / 2) + 30; 
                
                ctx.fillStyle = '#e0e0e0';
                ctx.fillText(text, textX, textY);
                ctx.save();
            }
        }]
    });
}


// 6. INITIAL SETUP



document.addEventListener('DOMContentLoaded', () => {

    predictBtn.disabled = true;

    
});