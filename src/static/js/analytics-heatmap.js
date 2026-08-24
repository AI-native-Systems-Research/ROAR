// Heatmap analytics visualization

let heatmapData = null;
let selectedMetric = null;

function renderHeatmap(data) {
    const analyticsContent = document.getElementById('analyticsContent');
    heatmapData = data;

    if (data.generators.length === 0 || data.problems.length === 0) {
        analyticsContent.innerHTML = '<div class="empty-state">No data available for heatmap.</div>';
        return;
    }

    // Default to combined_score if available, otherwise first metric
    if (!selectedMetric || !data.metrics.includes(selectedMetric)) {
        selectedMetric = data.metrics.includes('combined_score') ? 'combined_score' : data.metrics[0];
    }

    let html = `
        <div class="heatmap-controls">
            <label for="metricSelect">Metric:</label>
            <select id="metricSelect" onchange="updateHeatmap()">
                ${data.metrics.map(m => `<option value="${m}" ${m === selectedMetric ? 'selected' : ''}>${m}</option>`).join('')}
            </select>
        </div>
        <div class="heatmap-container">
            <table class="heatmap-table">
                <thead>
                    <tr>
                        <th>Generator</th>
                        ${data.problems.map(p => `<th class="problem-header" title="${escapeHtml(p.label)}">${escapeHtml(p.label)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;

    // Find min/max for color scaling
    let minVal = Infinity, maxVal = -Infinity;
    for (const gen of data.generators) {
        for (const prob of data.problems) {
            const val = data.matrix[gen]?.[prob.cluster_id]?.[selectedMetric];
            if (val !== undefined) {
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
            }
        }
    }

    for (const gen of data.generators) {
        html += `<tr><td class="generator-cell" title="${escapeHtml(gen)}">${escapeHtml(gen)}</td>`;
        for (const prob of data.problems) {
            const val = data.matrix[gen]?.[prob.cluster_id]?.[selectedMetric];
            if (val !== undefined) {
                const normalized = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0.5;
                const color = getHeatmapColor(normalized);
                const textColor = normalized > 0.5 ? '#000' : '#fff';
                html += `<td class="heatmap-cell" style="background-color: ${color}; color: ${textColor};">${val.toFixed(1)}</td>`;
            } else {
                html += `<td class="heatmap-cell heatmap-empty">-</td>`;
            }
        }
        html += '</tr>';
    }

    html += '</tbody></table></div>';
    analyticsContent.innerHTML = html;
}

function updateHeatmap() {
    selectedMetric = document.getElementById('metricSelect').value;
    if (heatmapData) {
        renderHeatmap(heatmapData);
    }
}

function getHeatmapColor(normalized) {
    // Red (low) to yellow (mid) to green (high) gradient
    if (normalized < 0.5) {
        // Red to yellow
        const t = normalized * 2;
        const r = 239;
        const g = Math.round(68 + t * (179 - 68));
        const b = Math.round(68 + t * (71 - 68));
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // Yellow to green
        const t = (normalized - 0.5) * 2;
        const r = Math.round(239 - t * (239 - 34));
        const g = Math.round(179 + t * (197 - 179));
        const b = Math.round(71 - t * (71 - 94));
        return `rgb(${r}, ${g}, ${b})`;
    }
}
