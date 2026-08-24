// Variance analytics visualization (Q1: Basin Structure / Multimodality)

let varianceGroupBy = 'aggregate';  // 'aggregate', 'model', 'algorithm', 'model_algorithm'
let varianceProblems = [];  // [] means no problems selected
let availableVarianceProblems = [];
let varianceDirty = false;

function markVarianceDirty() {
    varianceDirty = true;
    updateVarianceApplyButton();
}

function updateVarianceApplyButton() {
    const btn = document.getElementById('varianceApplyBtn');
    if (btn) {
        btn.className = varianceDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = varianceDirty ? 'Apply *' : 'Apply';
    }
}

function renderVariance(data) {
    const analyticsContent = document.getElementById('analyticsContent');

    if (!data.cells || data.cells.length === 0) {
        analyticsContent.innerHTML = '<div class="empty-state">No variance data available. Need at least 2 runs per configuration.</div>';
        return;
    }

    const summary = data.summary;

    let html = `
        <div class="variance-summary">
            <div class="summary-card">
                <div class="summary-value">${summary.total_cells}</div>
                <div class="summary-label">Configurations</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">${summary.total_campaigns}</div>
                <div class="summary-label">Total Runs</div>
            </div>
            <div class="summary-card ${summary.high_bc_cells > 0 ? 'highlight' : ''}">
                <div class="summary-value">${summary.high_bc_cells}</div>
                <div class="summary-label">Bimodal (BC &gt; 0.55)</div>
            </div>
        </div>

        <div class="variance-description">
            <p><strong>Q1: Basin Structure / Multimodality</strong> - Do identical configurations produce multimodal final-score distributions?</p>
            <p>Bimodality coefficient BC = (γ² + 1) / (κ + 3). BC &gt; 0.55 suggests bimodal distribution (two distinct attractors).</p>
        </div>

        <div id="varianceContent">
            <div class="loading">Loading...</div>
        </div>
    `;

    analyticsContent.innerHTML = html;
    loadVarianceProblems();
}

async function loadVarianceProblems() {
    try {
        const res = await fetch('/analytics/variance/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableVarianceProblems = data.problems || [];
        renderVarianceControls();
    } catch (error) {
        const content = document.getElementById('varianceContent');
        content.innerHTML = `<div class="message error" style="display:block;">Error loading problems: ${error.message}</div>`;
    }
}

function renderVarianceControls() {
    const content = document.getElementById('varianceContent');

    const description = varianceGroupBy === 'aggregate'
        ? 'Violin plots showing score distributions per problem, aggregating all models and algorithms together.'
        : 'Violin plots showing score distributions for each configuration within selected problems.';

    content.innerHTML = `
        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="varianceGroupBySelect" onchange="markVarianceDirty()">
                    <option value="aggregate" ${varianceGroupBy === 'aggregate' ? 'selected' : ''}>Aggregate</option>
                    <option value="model" ${varianceGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="algorithm" ${varianceGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model_algorithm" ${varianceGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="varianceApplyBtn" class="btn-apply" onclick="applyVarianceChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadVarianceFigure()">Download</button>
            </div>
            <p class="section-description">${description}</p>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${availableVarianceProblems.map(p => `
                    <label style="margin-right: 10px;">
                        <input type="checkbox" id="var_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}"
                               onchange="markVarianceDirty()"
                               ${varianceProblems.includes(p) ? 'checked' : ''}>
                        ${escapeHtml(truncateVarianceProblem(p))}
                    </label>
                `).join('')}
            </div>
            <div class="figure-wrapper" id="varianceFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view variance data.</div>
            </div>
        </div>
    `;
}

function truncateVarianceProblem(problem) {
    if (problem.length > 40) {
        return problem.substring(0, 37) + '...';
    }
    return problem;
}

function getSelectedVarianceProblems() {
    const problems = [];
    for (const p of availableVarianceProblems) {
        const checkbox = document.getElementById(`var_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyVarianceChanges() {
    varianceGroupBy = document.getElementById('varianceGroupBySelect')?.value || varianceGroupBy;
    varianceDirty = false;
    updateVarianceApplyButton();
    updateVarianceFigures();
}

function updateVarianceFigures() {
    varianceProblems = getSelectedVarianceProblems();

    const wrapper = document.getElementById('varianceFigureWrapper');
    if (!wrapper) return;

    if (varianceProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view variance data.</div>';
        return;
    }

    // Always show one figure per selected problem
    let html = '';
    for (const problem of varianceProblems) {
        const figUrl = `/analytics/variance/figure/${encodeURIComponent(problem)}?group_by=${varianceGroupBy}`;
        html += `
            <div class="problem-figure-section" style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0;">${escapeHtml(truncateVarianceProblem(problem))}</h4>
                    <button class="btn-secondary btn-sm" onclick="downloadSingleVarianceFigure('${encodeURIComponent(problem)}')">Download</button>
                </div>
                <img src="${figUrl}" alt="${escapeHtml(problem)} Variance"
                     style="max-width: 100%;"
                     onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure for ${escapeHtml(truncateVarianceProblem(problem))}</div>'" />
            </div>
        `;
    }
    wrapper.innerHTML = html;
}

function downloadVarianceFigure() {
    if (varianceProblems.length === 0) return;

    // Download first selected problem
    const link = document.createElement('a');
    const problem = varianceProblems[0];
    link.href = `/analytics/variance/figure/${encodeURIComponent(problem)}?group_by=${varianceGroupBy}`;
    const problemName = problem.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30);
    link.download = `variance_q1_${problemName}_${varianceGroupBy}.png`;
    link.click();
}

function downloadSingleVarianceFigure(encodedProblem) {
    const link = document.createElement('a');
    link.href = `/analytics/variance/figure/${encodedProblem}?group_by=${varianceGroupBy}`;
    const problemName = decodeURIComponent(encodedProblem).replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30);
    link.download = `variance_q1_${problemName}_${varianceGroupBy}.png`;
    link.click();
}
