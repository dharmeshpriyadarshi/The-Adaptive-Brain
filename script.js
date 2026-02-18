document.addEventListener('DOMContentLoaded', () => {
    
    // DOM Elements
    const citySelect = document.getElementById('citySelect');
    const datePicker = document.getElementById('datePicker');
    const predictBtn = document.getElementById('predictBtn');
    const resultSection = document.getElementById('resultSection');
    const loadingDiv = document.getElementById('loading');
    
    // KPI Elements
    const aqiValue = document.getElementById('aqiValue');
    const aqiStatus = document.getElementById('aqiStatus');
    const trendName = document.getElementById('trendName');
    const trendCard = document.getElementById('trendCard');
    const aqiCard = document.getElementById('aqiCard');
    
    // Chart
    const ctx = document.getElementById('trendChart').getContext('2d');
    let trendChart;

    // State
    let predictionsData = null;

    // Fetch Predictions
    async function loadPredictions() {
        try {
            const response = await fetch('predictions.json');
            predictionsData = await response.json();
            console.log("Predictions loaded:", predictionsData);
        } catch (error) {
            console.error("Error loading predictions:", error);
            alert("Failed to load prediction model data.");
        }
    }

    // Initialize
    loadPredictions();

    // Event Listener
    predictBtn.addEventListener('click', () => {
        if (!predictionsData) return;
        
        const city = citySelect.value;
        const date = datePicker.value;
        
        handlePrediction(city, date);
    });

    function handlePrediction(city, date) {
        // Show loading
        resultSection.classList.add('hidden');
        loadingDiv.classList.remove('hidden');

        // Simulate calculation delay for effect
        setTimeout(() => {
            loadingDiv.classList.add('hidden');
            resultSection.classList.remove('hidden');
            
            updateUI(city, date);
        }, 800);
    }

    function updateUI(city, date) {
        // Get data for the city
        const cityData = predictionsData[city];
        
        if (!cityData) {
            alert(`No model data available for ${city} yet.`);
            return;
        }

        // Try to get exact date, or fallback (since we are using mock data mostly)
        // In real app, we would have data for every day.
        let dayData = cityData[date];
        
        // --- MOCK FALLBACK LOGIC FOR DEMO ---
        // If specific date not in mock, grab the first available one to show *something*
        if (!dayData) {
            const keys = Object.keys(cityData);
            dayData = cityData[keys[0]]; 
            console.log("Using fallback mock data for demo purposes");
        }
        // ------------------------------------

        // Update KPIs
        aqiValue.textContent = dayData.aqi;
        trendName.textContent = dayData.trend;
        
        // Status Logic
        let statusText = "Moderate";
        let statusClass = "status-poor";
        
        if (dayData.aqi <= 100) {
            statusText = "Satisfactory";
            statusClass = "status-good";
        } else if (dayData.aqi > 300) {
            statusText = "Severe";
            statusClass = "status-severe";
        } else { // 100-300
            statusText = "Poor";
            statusClass = "status-poor";
        }
        
        // Remove old classes
        aqiStatus.className = "kpi-status";
        aqiStatus.classList.add(statusClass);
        aqiStatus.textContent = statusText;
        
        // Anomaly Highlight
        if (dayData.is_anomaly) {
            trendCard.style.borderColor = "var(--accent-red)";
            trendName.style.color = "var(--accent-red)";
        } else {
            trendCard.style.borderColor = "var(--glass-border)";
            trendName.style.color = "var(--text-primary)";
        }

        // Render Chart
        renderChart(city, date, dayData.aqi);
    }

    function renderChart(city, targetDate, targetAQI) {
        // Destroy old chart if exists
        if (trendChart) {
            trendChart.destroy();
        }

        // Generate some mock trend data around the target AQI 
        // Real app would pull this from the JSON
        const labels = [];
        const dataPoints = [];
        
        // Generate +/- 5 days
        for (let i = -5; i <= 5; i++) {
            labels.push(`Day ${i > 0 ? '+' + i : i}`);
            // Add some random variance to the target AQI to make a line
            let val = targetAQI + (Math.random() * 50 - 25); 
            if (i === 0) val = targetAQI; // Exact match for center
            dataPoints.push(val);
        }

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Projected Pollution Trend (${city})`,
                    data: dataPoints,
                    borderColor: '#66ff99',
                    backgroundColor: 'rgba(102, 255, 153, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: (ctx) => {
                        const index = ctx.dataIndex;
                        return index === 5 ? 6 : 3; // Highlight center point
                    },
                    pointBackgroundColor: (ctx) => {
                        const index = ctx.dataIndex;
                        return index === 5 ? '#ffffff' : '#66ff99';
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#a0a09a' }
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#a0a09a' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a09a' }
                    }
                }
            }
        });
    }

});
