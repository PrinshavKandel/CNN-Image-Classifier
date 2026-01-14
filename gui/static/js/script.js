let currentChart = null;
let currentGraphIndex = 0;
const graphs = [
    {
        labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6', 'Epoch 7', 'Epoch 8', 'Epoch 9', 'Epoch 10'],
        values: [52.45, 80.23, 88.94, 93.31, 91.63, 94.24, 94.90, 95.23, 95.98, 95.77],
        title: 'Training Accuracy'
    },
   
   
];

// File upload handling
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    console.log("File selected:", file);
    
    if (file) {
        // Preview the image locally
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            console.log("Image preview loaded");
        };
        reader.readAsDataURL(file);

        // Upload to server
        const formData = new FormData();
        formData.append('image', file);
        
        console.log("Uploading file to server...");

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            console.log("Response status:", response.status);
            const data = await response.json();
            console.log("Response data:", data);
            
            if (data.success) {
                predictBtn.disabled = false;
                predictBtn.dataset.filename = data.filename;
                console.log("Upload successful, filename:", data.filename);
            } else {
                alert('Error uploading image: ' + data.error);
                console.error("Upload failed:", data.error);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Error uploading image: ' + error.message);
        }
    }
});

// Prediction handling - Single Model
predictBtn.addEventListener('click', async () => {
    const filename = predictBtn.dataset.filename;
    
    if (!filename) {
        alert('Please upload an image first');
        return;
    }

    document.getElementById('loading').classList.add('active');
    predictBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                filename: filename
            })
        });

        const data = await response.json();
        console.log("Prediction response:", data);
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Update metrics - check if elements exist first
        const predictionEl = document.getElementById('prediction');
        const accuracyEl = document.getElementById('accuracy');
        const precisionEl = document.getElementById('precision');
        const recallEl = document.getElementById('recall');
        const f1ScoreEl = document.getElementById('f1-score');
        
        if (predictionEl) predictionEl.textContent = data.prediction;
        if (accuracyEl) accuracyEl.textContent = data.accuracy;
        if (precisionEl) precisionEl.textContent = data.precision;
        if (recallEl) recallEl.textContent = data.recall;
        if (f1ScoreEl) f1ScoreEl.textContent = data.f1_score;
        
        // Color code prediction
        if (predictionEl) {
            if (data.prediction === 'TB Positive') {
                predictionEl.style.color = '#dc3545';
            } else {
                predictionEl.style.color = '#28a745';
            }
        }

        // Update gauge
        updateGauge(data.probability);
        
        // Update chart
        currentGraphIndex = 0;
        updateChart(currentGraphIndex);
        
        console.log("✅ All elements updated successfully");
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction: ' + error.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
        predictBtn.disabled = false;
    }
});

// Chart handling
function updateChart(index) {
    const ctx = document.getElementById('results-chart');
    const graphData = graphs[index];
    
    if (currentChart) {
        currentChart.destroy();
    }

    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: graphData.labels,
            datasets: [{
                label: graphData.title,
                data: graphData.values,
                backgroundColor: '#a8a8a8',
                borderColor: '#888',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        font: {
                            size: 12
                        }
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    document.querySelector('.graph-label').textContent = `  GNAA`;
}

// Gauge meter
function updateGauge(probability) {
    const canvas = document.getElementById('gauge-canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 320;
    canvas.height = 220;
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height - 40;
    const radius = 110;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 28;
    ctx.stroke();
    
    // Draw filled arc
    const endAngle = Math.PI + (Math.PI * probability);
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, endAngle);
    
    let color;
    if (probability < 0.4) {
        color = '#4CAF50';
    } else if (probability < 0.7) {
        color = '#FFC107';
    } else {
        color = '#F44336';
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 28;
    ctx.stroke();
    
    // Draw percentage
    ctx.fillStyle = '#000';
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(probability * 100) + '%', centerX, centerY - 15);
    
    // Draw needle
    const needleAngle = Math.PI + (Math.PI * probability);
    const needleLength = radius - 20;
    
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
        centerX + needleLength * Math.cos(needleAngle),
        centerY + needleLength * Math.sin(needleAngle)
    );
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 4;
    ctx.stroke();
    
    // Center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
    ctx.fillStyle = '#000';
    ctx.fill();
}

// Graph navigation
document.getElementById('prev-graph').addEventListener('click', () => {
    if (currentChart) {
        currentGraphIndex = (currentGraphIndex - 1 + graphs.length) % graphs.length;
        updateChart(currentGraphIndex);
    }
});

document.getElementById('next-graph').addEventListener('click', () => {
    if (currentChart) {
        currentGraphIndex = (currentGraphIndex + 1) % graphs.length;
        updateChart(currentGraphIndex);
    }
});

// Initialize gauge with 0
updateGauge(0);