"""
Web-based dashboard for monitoring SAE features during gameplay.

Provides real-time visualization of:
- Feature activations across all layers
- Alert history
- Suspicion scores over time
- Anomaly detection
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class Dashboard:
    """
    Web dashboard for SAE monitoring.

    Generates HTML/JS for real-time feature visualization.
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage for updates
        self.step_data: List[Dict[str, Any]] = []

    def update(self, step_num: int, monitoring_report, features: Dict):
        """Update dashboard with new step data."""
        # Convert features to serializable format
        feature_data = {}
        for layer, feat in features.items():
            # Get top 50 most active features
            top_vals, top_idx = feat.abs().topk(min(50, len(feat)))
            feature_data[layer] = {
                'top_indices': top_idx.cpu().tolist(),
                'top_values': top_vals.cpu().tolist()
            }

        step_record = {
            'step': step_num,
            'suspicion_score': monitoring_report.suspicion_score,
            'num_alerts': len(monitoring_report.new_alerts),
            'num_anomalies': len(monitoring_report.anomalies),
            'features': feature_data
        }

        self.step_data.append(step_record)

        # Save updated data
        self._save_data()

    def _save_data(self):
        """Save data to JSON for dashboard."""
        data_path = self.output_dir / 'dashboard_data.json'
        with open(data_path, 'w') as f:
            json.dump({
                'steps': self.step_data
            }, f)

    def generate_html(self):
        """Generate the dashboard HTML."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>SAE Guardian - Monitor Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            margin: 0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
        }

        .panel {
            background: #1a1a1a;
            border: 2px solid #00ff00;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }

        .panel h2 {
            margin-top: 0;
            color: #00ff00;
            font-size: 1.2em;
        }

        .metric {
            display: inline-block;
            margin: 10px 20px;
        }

        .metric-label {
            color: #888;
            font-size: 0.9em;
        }

        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
        }

        .alert {
            background: #ff000020;
            border-left: 3px solid #ff0000;
            padding: 10px;
            margin: 5px 0;
        }

        .alert.high {
            border-left-color: #ff0000;
        }

        .alert.medium {
            border-left-color: #ff8800;
        }

        .alert.low {
            border-left-color: #ffff00;
        }

        .chart-container {
            height: 200px;
            margin: 20px 0;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }

        .feature-item {
            background: #0a0a0a;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 3px;
        }

        .feature-bar {
            height: 20px;
            background: linear-gradient(90deg, #00ff00, #ff0000);
            border-radius: 3px;
            margin-top: 5px;
        }
    </style>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>⚠️ SAE GUARDIAN - MONITORING DASHBOARD ⚠️</h1>

        <div class="panel">
            <h2>CURRENT STATUS</h2>
            <div class="metric">
                <div class="metric-label">STEP</div>
                <div class="metric-value" id="current-step">0</div>
            </div>
            <div class="metric">
                <div class="metric-label">SUSPICION SCORE</div>
                <div class="metric-value" id="suspicion-score">0.00</div>
            </div>
            <div class="metric">
                <div class="metric-label">ALERTS</div>
                <div class="metric-value" id="alert-count">0</div>
            </div>
            <div class="metric">
                <div class="metric-label">ANOMALIES</div>
                <div class="metric-value" id="anomaly-count">0</div>
            </div>
        </div>

        <div class="panel">
            <h2>SUSPICION SCORE OVER TIME</h2>
            <div id="suspicion-chart" class="chart-container"></div>
        </div>

        <div class="panel">
            <h2>TOP ACTIVE FEATURES (Current Step)</h2>
            <div id="feature-display" class="feature-grid"></div>
        </div>

        <div class="panel">
            <h2>RECENT ALERTS</h2>
            <div id="alert-log"></div>
        </div>
    </div>

    <script>
        let dashboardData = [];
        let currentStep = 0;

        async function loadData() {
            try {
                const response = await fetch('dashboard_data.json');
                const data = await response.json();
                dashboardData = data.steps;
                updateDashboard();
            } catch (e) {
                console.error('Error loading data:', e);
            }
        }

        function updateDashboard() {
            if (dashboardData.length === 0) return;

            currentStep = dashboardData.length - 1;
            const current = dashboardData[currentStep];

            // Update metrics
            document.getElementById('current-step').textContent = current.step;
            document.getElementById('suspicion-score').textContent = current.suspicion_score.toFixed(2);
            document.getElementById('alert-count').textContent = current.num_alerts;
            document.getElementById('anomaly-count').textContent = current.num_anomalies;

            // Update suspicion chart
            const steps = dashboardData.map(d => d.step);
            const scores = dashboardData.map(d => d.suspicion_score);

            const chartData = [{
                x: steps,
                y: scores,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#00ff00', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 255, 0, 0.1)'
            }];

            const layout = {
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#0a0a0a',
                font: { color: '#00ff00', family: 'Courier New' },
                xaxis: { title: 'Step', gridcolor: '#333' },
                yaxis: { title: 'Suspicion Score', range: [0, 1], gridcolor: '#333' },
                margin: { t: 20, b: 40, l: 50, r: 20 }
            };

            Plotly.newPlot('suspicion-chart', chartData, layout, { responsive: true });

            // Update feature display
            updateFeatureDisplay(current.features);
        }

        function updateFeatureDisplay(features) {
            const container = document.getElementById('feature-display');
            container.innerHTML = '';

            for (const [layer, data] of Object.entries(features)) {
                const topFeatures = data.top_indices.slice(0, 10);
                const topValues = data.top_values.slice(0, 10);

                topFeatures.forEach((idx, i) => {
                    const item = document.createElement('div');
                    item.className = 'feature-item';

                    const value = topValues[i];
                    const normalized = Math.min(value / 5.0, 1.0);

                    item.innerHTML = `
                        <div><strong>${layer.split('.')[1]}.${idx}</strong></div>
                        <div style="font-size: 0.9em; color: #888;">Activation: ${value.toFixed(2)}</div>
                        <div class="feature-bar">
                            <div style="width: ${normalized * 100}%; height: 100%; background: #00ff00;"></div>
                        </div>
                    `;

                    container.appendChild(item);
                });
            }
        }

        // Auto-refresh every 2 seconds
        setInterval(loadData, 2000);
        loadData();
    </script>
</body>
</html>
"""

        html_path = self.output_dir / 'dashboard.html'
        with open(html_path, 'w') as f:
            f.write(html)

        print(f"Dashboard generated: {html_path}")
        print(f"Open in browser: file://{html_path.absolute()}")

        return html_path
