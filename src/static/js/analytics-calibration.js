// Calibration analytics visualization (Q2: Stagnation Heuristic Calibration)

let calibrationGroupBy = 'aggregate';  // 'aggregate', 'model', 'algorithm', 'model_algorithm'
let calibrationProblems = [];  // [] means no problems selected
let availableCalibrationProblems = [];
let calibrationDirty = false;

function markCalibrationDirty() {
    calibrationDirty = true;
    updateCalibrationApplyButton();
}

function updateCalibrationApplyButton() {
    const btn = document.getElementById('calibrationApplyBtn');
    if (btn) {
        btn.className = calibrationDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = calibrationDirty ? 'Apply *' : 'Apply';
    }
}

function renderCalibration(data) {
    const analyticsContent = document.getElementById('analyticsContent');

    if (!data.by_threshold || Object.keys(data.by_threshold).length === 0) {
        analyticsContent.innerHTML = '<div class="empty-state">No calibration data available. Need BoN campaigns with sufficient iterations.</div>';
        return;
    }

    const overall = data.overall;
    const escapeRatePct = overall.escape_rate ? (overall.escape_rate * 100).toFixed(1) : 'N/A';
    const direction = overall.escape_rate < 0.63 ? 'too optimistic' : 'too pessimistic';
    const directionClass = overall.escape_rate < 0.63 ? 'cv-high' : 'cv-low';

    let html = `
        <div class="variance-summary">
            <div class="summary-card">
                <div class="summary-value">${overall.total_observations}</div>
                <div class="summary-label">Total Observations</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">${overall.uncensored}</div>
                <div class="summary-label">Completed Windows</div>
            </div>
            <div class="summary-card ${overall.escape_rate < 0.4 ? 'highlight' : ''}">
                <div class="summary-value">${escapeRatePct}%</div>
                <div class="summary-label">Overall Escape Rate</div>
            </div>
        </div>

        <div class="variance-description">
            <p><strong>Q2: Rule-of-3 Calibration Test</strong> - Is the stagnation heuristic calibrated for BoN runs?</p>
            <p>After k non-improving iterations, the rule of 3 predicts escape within k/3 more iterations with ≤63% probability.
               Escape rate <strong>&lt; 63%</strong> = conservative (actual stagnation is worse than heuristic implies).
               <strong>&gt; 63%</strong> = anti-conservative (runs escape faster than predicted).</p>
            <p class="${directionClass}" style="padding: 8px; border-radius: 4px; margin-top: 8px;">
                <strong>Finding:</strong> The heuristic is <strong>${direction}</strong> —
                ${direction === 'too optimistic'
                    ? 'runs are far more stuck than the rule of 3 predicts. History-driven context correlation reduces effective trial count.'
                    : 'runs escape faster than the rule of 3 predicts. The generator adapts favorably from accumulated failures.'}
            </p>
        </div>

        <div id="calibrationContent">
            <div class="loading">Loading...</div>
        </div>
    `;

    analyticsContent.innerHTML = html;
    loadCalibrationProblems();
}

async function loadCalibrationProblems() {
    try {
        const res = await fetch('/analytics/calibration/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableCalibrationProblems = data.problems || [];
        renderCalibrationControls();
    } catch (error) {
        const content = document.getElementById('calibrationContent');
        content.innerHTML = `<div class="message error" style="display:block;">Error loading problems: ${error.message}</div>`;
    }
}

function renderCalibrationControls() {
    const content = document.getElementById('calibrationContent');

    const description = calibrationGroupBy === 'aggregate'
        ? 'Bar chart showing escape rates at different stagnation thresholds (k≥10, k≥20, k≥30, k≥40) compared to the 63% rule-of-3 reference line.'
        : 'Line plot showing how escape probability changes with stagnation length for each selected problem.';

    content.innerHTML = `
        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="calibrationGroupBySelect" onchange="markCalibrationDirty()">
                    <option value="aggregate" ${calibrationGroupBy === 'aggregate' ? 'selected' : ''}>Aggregate</option>
                    <option value="model" ${calibrationGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="algorithm" ${calibrationGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model_algorithm" ${calibrationGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="calibrationApplyBtn" class="btn-apply" onclick="applyCalibrationChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCalibrationFigure()">Download</button>
            </div>
            <p class="section-description">${description}</p>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${availableCalibrationProblems.map(p => `
                    <label style="margin-right: 10px;">
                        <input type="checkbox" id="cal_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}"
                               onchange="markCalibrationDirty()"
                               ${calibrationProblems.includes(p) ? 'checked' : ''}>
                        ${escapeHtml(p)}
                    </label>
                `).join('')}
            </div>
            <div class="figure-wrapper" id="calibrationFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view calibration data.</div>
            </div>
        </div>
    `;
}

function getSelectedCalibrationProblems() {
    const problems = [];
    for (const p of availableCalibrationProblems) {
        const checkbox = document.getElementById(`cal_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCalibrationChanges() {
    calibrationGroupBy = document.getElementById('calibrationGroupBySelect')?.value || calibrationGroupBy;
    calibrationDirty = false;
    updateCalibrationApplyButton();
    updateCalibrationFigures();
}

function updateCalibrationFigures() {
    calibrationProblems = getSelectedCalibrationProblems();

    const wrapper = document.getElementById('calibrationFigureWrapper');
    if (!wrapper) return;

    if (calibrationProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view calibration data.</div>';
        return;
    }

    const problemsParam = encodeURIComponent(calibrationProblems.join(','));

    if (calibrationGroupBy === 'aggregate') {
        // Single aggregate bar chart with selected problems on x-axis
        const figUrl = `/analytics/calibration/aggregate/figure?problems=${problemsParam}`;
        wrapper.innerHTML = `
            <img id="calibrationFigure" src="${figUrl}" alt="Aggregate Calibration"
                 onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure</div>'" />
        `;
    } else {
        // Single line plot aggregating all selected problems
        const figUrl = `/analytics/calibration/figure?problems=${problemsParam}&group_by=${calibrationGroupBy}`;
        wrapper.innerHTML = `
            <img id="calibrationFigure" src="${figUrl}" alt="Calibration by ${calibrationGroupBy}"
                 style="max-width: 100%;"
                 onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure</div>'" />
        `;
    }
}

function downloadCalibrationFigure() {
    if (calibrationProblems.length === 0) return;

    const link = document.createElement('a');
    const problemsParam = encodeURIComponent(calibrationProblems.join(','));
    const problemSuffix = calibrationProblems.length < 3
        ? `_${calibrationProblems.join('_').toLowerCase()}`
        : '';

    if (calibrationGroupBy === 'aggregate') {
        link.href = `/analytics/calibration/aggregate/figure?problems=${problemsParam}`;
        link.download = `calibration_q2_aggregate${problemSuffix}.png`;
    } else {
        link.href = `/analytics/calibration/figure?problems=${problemsParam}&group_by=${calibrationGroupBy}`;
        link.download = `calibration_q2_${calibrationGroupBy}${problemSuffix}.png`;
    }
    link.click();
}

