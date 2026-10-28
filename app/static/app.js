// Safe DOM manipulation helpers using textContent and createElement
function createEl(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs)) {
    if (key === "className") {
      el.className = val;
    } else if (key === "textContent") {
      el.textContent = val;
    } else if (key === "disabled") {
      if (val) el.setAttribute("disabled", "");
      else el.removeAttribute("disabled");
    } else {
      el.setAttribute(key, val);
    }
  }
  for (const child of children) {
    if (child == null) continue;
    if (typeof child === "string" || typeof child === "number") {
      el.appendChild(document.createTextNode(String(child)));
    } else if (child instanceof Node) {
      el.appendChild(child);
    }
  }
  return el;
}

function clearEl(el) {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

// Fixed status labels with symbols (never colour alone)
const STATUS_CONFIG = {
  HIGH: { label: "▲ Above range", className: "status-high" },
  LOW: { label: "▼ Below range", className: "status-low" },
  NORMAL: { label: "● Within range", className: "status-normal" },
  UNKNOWN: { label: "? Not compared", className: "status-unknown" },
  CRITICAL_HIGH: { label: "‼ Critical high", className: "status-critical-high" },
  CRITICAL_LOW: { label: "‼ Critical low", className: "status-critical-low" }
};

const VERIFY_TEXT = "Please verify against your original report";

// Global application state
const state = {
  activeProfileId: 1,
  profiles: [],
  reports: [],
  currentReport: null,
  currentExplanation: null,
  currentQuestions: null,
  latestTests: {},
  selectedTrendTest: null,
  trendData: []
};

// Unified API fetch helper with relative URLs
async function apiRequest(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    const netErr = new Error("Could not reach the app. Is it still running?");
    netErr.isNetworkError = true;
    throw netErr;
  }

  let data = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      data = await response.json();
    } catch (e) {
      data = null;
    }
  } else {
    try {
      const text = await response.text();
      data = { detail: text };
    } catch (e) {
      data = null;
    }
  }

  if (!response.ok) {
    let message = null;
    if (data && typeof data.detail === "string") {
      message = data.detail;
    } else if (data && Array.isArray(data.detail)) {
      message = data.detail.map(d => d.msg || JSON.stringify(d)).join("; ");
    }
    const error = new Error(message || `Request failed with status ${response.status}`);
    error.status = response.status;
    error.data = data;
    throw error;
  }

  return data;
}

// Live region announcements
function announceLive(msg) {
  const region = document.getElementById("aria-status-region");
  if (region) {
    region.textContent = "";
    setTimeout(() => { region.textContent = msg; }, 50);
  }
}

// Toast notification
let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById("toast-notification");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove("hidden");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.add("hidden");
  }, 4000);
}

// Dynamic Plotly loader from self-hosted vendor route on first need
let plotlyPromise = null;
function loadPlotly() {
  if (window.Plotly) {
    return Promise.resolve(window.Plotly);
  }
  if (!plotlyPromise) {
    plotlyPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/vendor/plotly.min.js";
      script.onload = () => resolve(window.Plotly);
      script.onerror = () => reject(new Error("Failed to load plotly script"));
      document.head.appendChild(script);
    });
  }
  return plotlyPromise;
}

// Plotly marker shapes per flag
function getPlotlyShapeForFlag(flag) {
  switch (flag) {
    case "HIGH":
      return "triangle-up";
    case "LOW":
      return "triangle-down";
    case "CRITICAL_HIGH":
      return "star";
    case "CRITICAL_LOW":
      return "cross";
    case "NORMAL":
      return "circle";
    case "UNKNOWN":
    default:
      return "square";
  }
}

// Format "in words" parenthetical
function formatWithWords(written, plain) {
  const w = (written || "").trim();
  const p = (plain || "").trim();
  if (!w && !p) return "-";
  if (!p || w.toLowerCase() === p.toLowerCase()) {
    return w || p;
  }
  if (!w) return p;
  return `${w} (${p})`;
}

// Format date with fallback
function formatDateString(val) {
  if (!val) return "Date not found";
  return val;
}

// Format number nicely
function formatNumericValue(val) {
  if (val == null) return "";
  return String(val);
}

// DOM Elements
let profileSelect;
let newProfileInput;
let addProfileForm;
let deleteProfileDataBtn;
let deleteConfirmBox;
let confirmDeleteDataBtn;
let cancelDeleteDataBtn;

let tabReportsBtn;
let tabPrescriptionBtn;
let tabReportsPanel;
let tabPrescriptionPanel;

let reportDropZone;
let reportFileInput;
let uploadReportBtn;
let reportSelectedFilename;
let uploadLoadingStatus;
let uploadStatusText;
let uploadErrorBox;
let uploadErrorText;
let openExistingReportBtn;

let reportsListContainer;
let refreshReportsBtn;

let reportViewContainer;
let reportMainHeading;
let reportDateValue;
let reportLabValue;
let reportTotalCount;
let reportAbnormalCount;
let reportExtractionBadge;
let criticalAlertBanner;
let criticalBannerMessage;
let reportWarningsContainer;
let reportWarningsList;
let resultsTableBody;

let explanationSection;
let explanationLangSelect;
let explanationDraftNotice;
let explanationNotice;
let explanationItemsList;
let explanationDisclaimer;

// Session-only explanation language state variable (never stored in localStorage or cookies)
// Supported languages: English, हिन्दी, తెలుగు
// Selector label: Explanation language
// Hint: Doctor questions are shown in English so you can share them with your doctor.
let sessionExplanationLang = "en";

let copyQuestionsBtn;
let copyStatusMsg;
let doctorQuestionsOrdered;
let trendNotesWrapper;
let doctorTrendNotesList;
let generalQuestionsWrapper;
let doctorGeneralQuestionsList;

let trendTestSelect;
let trendSinglePointNote;
let trendPlot;
let trendDataTableBody;
let printSummaryBtn;
let printDateLine;

let prescriptionDropZone;
let prescriptionFileInput;
let uploadPrescriptionBtn;
let prescriptionSelectedFilename;
let prescriptionLoadingStatus;
let prescriptionErrorBox;
let prescriptionErrorText;
let prescriptionViewContainer;
let prescriptionHeading;
let medicationsCardList;
let prescriptionOtherLinesSection;
let prescriptionOtherLinesList;
let prescriptionWarningsSection;
let prescriptionWarningsList;
let prescriptionDisclaimerText;

let selectedReportFile = null;
let selectedPrescriptionFile = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initDomReferences();
  initEventHandlers();
  loadProfiles();
});

function initDomReferences() {
  profileSelect = document.getElementById("profile-select");
  newProfileInput = document.getElementById("new-profile-input");
  addProfileForm = document.getElementById("add-profile-form");
  deleteProfileDataBtn = document.getElementById("delete-profile-data-btn");
  deleteConfirmBox = document.getElementById("delete-confirm-box");
  confirmDeleteDataBtn = document.getElementById("confirm-delete-data-btn");
  cancelDeleteDataBtn = document.getElementById("cancel-delete-data-btn");

  tabReportsBtn = document.getElementById("tab-reports-btn");
  tabPrescriptionBtn = document.getElementById("tab-prescription-btn");
  tabReportsPanel = document.getElementById("tab-reports");
  tabPrescriptionPanel = document.getElementById("tab-prescription");

  reportDropZone = document.getElementById("report-drop-zone");
  reportFileInput = document.getElementById("report-file-input");
  uploadReportBtn = document.getElementById("upload-report-btn");
  reportSelectedFilename = document.getElementById("report-selected-filename");
  uploadLoadingStatus = document.getElementById("upload-loading-status");
  uploadStatusText = document.getElementById("upload-status-text");
  uploadErrorBox = document.getElementById("upload-error-box");
  uploadErrorText = document.getElementById("upload-error-text");
  openExistingReportBtn = document.getElementById("open-existing-report-btn");

  reportsListContainer = document.getElementById("reports-list-container");
  refreshReportsBtn = document.getElementById("refresh-reports-btn");

  reportViewContainer = document.getElementById("report-view-container");
  reportMainHeading = document.getElementById("report-main-heading");
  reportDateValue = document.getElementById("report-date-value");
  reportLabValue = document.getElementById("report-lab-value");
  reportTotalCount = document.getElementById("report-total-count");
  reportAbnormalCount = document.getElementById("report-abnormal-count");
  reportExtractionBadge = document.getElementById("report-extraction-badge");
  criticalAlertBanner = document.getElementById("critical-alert-banner");
  criticalBannerMessage = document.getElementById("critical-banner-message");
  reportWarningsContainer = document.getElementById("report-warnings-container");
  reportWarningsList = document.getElementById("report-warnings-list");
  resultsTableBody = document.getElementById("results-table-body");

  explanationSection = document.getElementById("explanation-section");
  explanationLangSelect = document.getElementById("explanation-lang-select");
  explanationDraftNotice = document.getElementById("explanation-draft-notice");
  explanationNotice = document.getElementById("explanation-overall-notice");
  explanationItemsList = document.getElementById("explanation-items-list");
  explanationDisclaimer = document.getElementById("explanation-disclaimer");

  copyQuestionsBtn = document.getElementById("copy-questions-btn");
  copyStatusMsg = document.getElementById("copy-status-msg");
  doctorQuestionsOrdered = document.getElementById("doctor-questions-ordered");
  trendNotesWrapper = document.getElementById("trend-notes-wrapper");
  doctorTrendNotesList = document.getElementById("doctor-trend-notes-list");
  generalQuestionsWrapper = document.getElementById("general-questions-wrapper");
  doctorGeneralQuestionsList = document.getElementById("doctor-general-questions-list");

  trendTestSelect = document.getElementById("trend-test-select");
  trendSinglePointNote = document.getElementById("trend-single-point-note");
  trendPlot = document.getElementById("trend-plot");
  trendDataTableBody = document.getElementById("trend-data-table-body");
  printSummaryBtn = document.getElementById("print-summary-btn");
  printDateLine = document.getElementById("print-date-line");

  prescriptionDropZone = document.getElementById("prescription-drop-zone");
  prescriptionFileInput = document.getElementById("prescription-file-input");
  uploadPrescriptionBtn = document.getElementById("upload-prescription-btn");
  prescriptionSelectedFilename = document.getElementById("prescription-selected-filename");
  prescriptionLoadingStatus = document.getElementById("prescription-loading-status");
  prescriptionErrorBox = document.getElementById("prescription-error-box");
  prescriptionErrorText = document.getElementById("prescription-error-text");
  prescriptionViewContainer = document.getElementById("prescription-view-container");
  prescriptionHeading = document.getElementById("prescription-heading");
  medicationsCardList = document.getElementById("medications-card-list");
  prescriptionOtherLinesSection = document.getElementById("prescription-other-lines-section");
  prescriptionOtherLinesList = document.getElementById("prescription-other-lines-list");
  prescriptionWarningsSection = document.getElementById("prescription-warnings-section");
  prescriptionWarningsList = document.getElementById("prescription-warnings-list");
  prescriptionDisclaimerText = document.getElementById("prescription-disclaimer-text");
}

function initEventHandlers() {
  // Explanation Language Selector Change
  if (explanationLangSelect) {
    explanationLangSelect.addEventListener("change", async (e) => {
      sessionExplanationLang = e.target.value;
      if (!state.currentReport) return;
      try {
        announceLive("Updating explanation language...");
        const exp = await apiRequest(`/api/reports/${state.currentReport.report_id}/explanation?lang=${sessionExplanationLang}`);
        state.currentExplanation = exp;
        renderExplanation(exp);
        announceLive("Explanation updated.");
      } catch (err) {
        showToast(err.message);
      }
    });
  }

  // Profile Selector Change
  profileSelect.addEventListener("change", (e) => {
    state.activeProfileId = parseInt(e.target.value, 10);
    hideReportView();
    loadReportsForProfile(state.activeProfileId);
  });

  // Add Profile Form
  addProfileForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const label = newProfileInput.value.trim();
    if (!label || label.length < 1 || label.length > 40) {
      showToast("Label must be between 1 and 40 characters.");
      return;
    }
    try {
      const newProf = await apiRequest("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label })
      });
      newProfileInput.value = "";
      showToast(`Profile "${newProf.label}" created.`);
      await loadProfiles(newProf.id);
    } catch (err) {
      showToast(err.message);
    }
  });

  // Delete Profile Data Confirmation Flow
  deleteProfileDataBtn.addEventListener("click", () => {
    deleteConfirmBox.classList.remove("hidden");
  });

  cancelDeleteDataBtn.addEventListener("click", () => {
    deleteConfirmBox.classList.add("hidden");
  });

  confirmDeleteDataBtn.addEventListener("click", async () => {
    try {
      await apiRequest(`/api/profiles/${state.activeProfileId}/data`, {
        method: "DELETE"
      });
      deleteConfirmBox.classList.add("hidden");
      showToast("All data for this profile has been deleted.");
      hideReportView();
      loadReportsForProfile(state.activeProfileId);
    } catch (err) {
      showToast(err.message);
    }
  });

  // Accessible Tab Navigation
  tabReportsBtn.addEventListener("click", () => switchTab("reports"));
  tabPrescriptionBtn.addEventListener("click", () => switchTab("prescription"));

  tabReportsBtn.addEventListener("keydown", handleTabKeydown);
  tabPrescriptionBtn.addEventListener("keydown", handleTabKeydown);

  // File Picker and Drag & Drop for Lab Reports
  reportFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleReportFileSelection(e.target.files[0]);
    }
  });

  setupDropZone(reportDropZone, (file) => handleReportFileSelection(file));

  uploadReportBtn.addEventListener("click", uploadSelectedReport);
  refreshReportsBtn.addEventListener("click", () => loadReportsForProfile(state.activeProfileId));

  // Copy Doctor Questions to Clipboard
  copyQuestionsBtn.addEventListener("click", copyDoctorQuestions);

  // Trend Test Selector
  trendTestSelect.addEventListener("change", (e) => {
    state.selectedTrendTest = e.target.value;
    renderTrendForSelectedTest();
  });

  // Print Summary Button
  printSummaryBtn.addEventListener("click", () => {
    const today = new Date().toISOString().split("T")[0];
    printDateLine.textContent = `Printed on ${today}`;
    window.print();
  });

  // Prescription Drag & Drop and File Picker
  prescriptionFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handlePrescriptionFileSelection(e.target.files[0]);
    }
  });

  setupDropZone(prescriptionDropZone, (file) => handlePrescriptionFileSelection(file));
  uploadPrescriptionBtn.addEventListener("click", uploadSelectedPrescription);
}

// Tab Switching
function switchTab(target) {
  if (target === "reports") {
    tabReportsBtn.classList.add("tab-active");
    tabReportsBtn.setAttribute("aria-selected", "true");
    tabReportsBtn.setAttribute("tabindex", "0");

    tabPrescriptionBtn.classList.remove("tab-active");
    tabPrescriptionBtn.setAttribute("aria-selected", "false");
    tabPrescriptionBtn.setAttribute("tabindex", "-1");

    tabReportsPanel.classList.remove("hidden");
    tabPrescriptionPanel.classList.add("hidden");
  } else {
    tabPrescriptionBtn.classList.add("tab-active");
    tabPrescriptionBtn.setAttribute("aria-selected", "true");
    tabPrescriptionBtn.setAttribute("tabindex", "0");

    tabReportsBtn.classList.remove("tab-active");
    tabReportsBtn.setAttribute("aria-selected", "false");
    tabReportsBtn.setAttribute("tabindex", "-1");

    tabPrescriptionPanel.classList.remove("hidden");
    tabReportsPanel.classList.add("hidden");
  }
}

function handleTabKeydown(e) {
  if (e.key === "ArrowRight" || e.key === "ArrowDown") {
    e.preventDefault();
    if (e.target === tabReportsBtn) {
      tabPrescriptionBtn.focus();
      switchTab("prescription");
    } else {
      tabReportsBtn.focus();
      switchTab("reports");
    }
  } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
    e.preventDefault();
    if (e.target === tabPrescriptionBtn) {
      tabReportsBtn.focus();
      switchTab("reports");
    } else {
      tabPrescriptionBtn.focus();
      switchTab("prescription");
    }
  }
}

// Generic Drag & Drop setup
function setupDropZone(dropZone, fileCallback) {
  ["dragenter", "dragover"].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      fileCallback(e.dataTransfer.files[0]);
    }
  });

  // Clicking drop zone triggers hidden file input
  dropZone.addEventListener("click", (e) => {
    if (e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") {
      const input = dropZone.querySelector("input[type=file]");
      if (input) input.click();
    }
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const input = dropZone.querySelector("input[type=file]");
      if (input) input.click();
    }
  });
}

// File handling for lab reports
function handleReportFileSelection(file) {
  uploadErrorBox.classList.add("hidden");
  openExistingReportBtn.classList.add("hidden");

  // Client-side convenience checks (server is final authority)
  const allowed = [".pdf", ".jpg", ".jpeg", ".png"];
  const fileName = file.name.toLowerCase();
  const hasValidExt = allowed.some(ext => fileName.endsWith(ext));

  if (!hasValidExt) {
    showUploadError("Unsupported file type. Please choose a PDF, PNG, or JPEG file.");
    selectedReportFile = null;
    uploadReportBtn.disabled = true;
    reportSelectedFilename.classList.add("hidden");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showUploadError("File exceeds 10 MB limit.");
    selectedReportFile = null;
    uploadReportBtn.disabled = true;
    reportSelectedFilename.classList.add("hidden");
    return;
  }

  selectedReportFile = file;
  reportSelectedFilename.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  reportSelectedFilename.classList.remove("hidden");
  uploadReportBtn.disabled = false;
}

function showUploadError(msg, existingId = null) {
  uploadErrorText.textContent = msg;
  uploadErrorBox.classList.remove("hidden");

  if (existingId) {
    openExistingReportBtn.classList.remove("hidden");
    openExistingReportBtn.onclick = () => {
      loadReport(existingId);
    };
  } else {
    openExistingReportBtn.classList.add("hidden");
  }
}

async function uploadSelectedReport() {
  if (!selectedReportFile) return;

  uploadErrorBox.classList.add("hidden");
  openExistingReportBtn.classList.add("hidden");
  uploadReportBtn.disabled = true;
  uploadLoadingStatus.classList.remove("hidden");
  uploadStatusText.textContent = "Reading your report...";
  announceLive("Reading your report...");

  const formData = new FormData();
  formData.append("file", selectedReportFile);
  formData.append("profile_id", String(state.activeProfileId));

  try {
    const uploadRes = await apiRequest("/api/reports/upload", {
      method: "POST",
      body: formData
    });

    uploadLoadingStatus.classList.add("hidden");
    uploadReportBtn.disabled = false;
    showToast("Report uploaded and analyzed successfully.");
    selectedReportFile = null;
    reportFileInput.value = "";
    reportSelectedFilename.classList.add("hidden");

    await loadReportsForProfile(state.activeProfileId);
    loadReport(uploadRes.report_id, uploadRes);
  } catch (err) {
    uploadLoadingStatus.classList.add("hidden");
    uploadReportBtn.disabled = false;

    if (err.status === 409 && err.data && err.data.existing_report_id) {
      showUploadError(err.message, err.data.existing_report_id);
    } else {
      showUploadError(err.message);
    }
  }
}

// File handling for prescriptions
function handlePrescriptionFileSelection(file) {
  prescriptionErrorBox.classList.add("hidden");

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    prescriptionErrorText.textContent = "Only PDF prescriptions are supported.";
    prescriptionErrorBox.classList.remove("hidden");
    selectedPrescriptionFile = null;
    uploadPrescriptionBtn.disabled = true;
    prescriptionSelectedFilename.classList.add("hidden");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    prescriptionErrorText.textContent = "File exceeds 10 MB limit.";
    prescriptionErrorBox.classList.remove("hidden");
    selectedPrescriptionFile = null;
    uploadPrescriptionBtn.disabled = true;
    prescriptionSelectedFilename.classList.add("hidden");
    return;
  }

  selectedPrescriptionFile = file;
  prescriptionSelectedFilename.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  prescriptionSelectedFilename.classList.remove("hidden");
  uploadPrescriptionBtn.disabled = false;
}

async function uploadSelectedPrescription() {
  if (!selectedPrescriptionFile) return;

  prescriptionErrorBox.classList.add("hidden");
  uploadPrescriptionBtn.disabled = true;
  prescriptionLoadingStatus.classList.remove("hidden");
  announceLive("Reading your prescription...");

  const formData = new FormData();
  formData.append("file", selectedPrescriptionFile);

  try {
    const summary = await apiRequest("/api/prescriptions/parse", {
      method: "POST",
      body: formData
    });

    prescriptionLoadingStatus.classList.add("hidden");
    uploadPrescriptionBtn.disabled = false;
    selectedPrescriptionFile = null;
    prescriptionFileInput.value = "";
    prescriptionSelectedFilename.classList.add("hidden");

    renderPrescriptionSummary(summary);
  } catch (err) {
    prescriptionLoadingStatus.classList.add("hidden");
    uploadPrescriptionBtn.disabled = false;
    prescriptionErrorText.textContent = err.message;
    prescriptionErrorBox.classList.remove("hidden");
  }
}

// Profiles Management
async function loadProfiles(selectId = null) {
  try {
    const profiles = await apiRequest("/api/profiles");
    state.profiles = profiles;
    clearEl(profileSelect);

    profiles.forEach(p => {
      const opt = createEl("option", { value: String(p.id), textContent: p.label });
      profileSelect.appendChild(opt);
    });

    if (selectId && profiles.some(p => p.id === selectId)) {
      state.activeProfileId = selectId;
    } else if (profiles.length > 0 && !profiles.some(p => p.id === state.activeProfileId)) {
      state.activeProfileId = profiles[0].id;
    }

    profileSelect.value = String(state.activeProfileId);
    await loadReportsForProfile(state.activeProfileId);
  } catch (err) {
    showToast(err.message);
  }
}

// Reports List for Active Profile
async function loadReportsForProfile(profileId) {
  try {
    const reports = await apiRequest(`/api/profiles/${profileId}/reports`);
    state.reports = reports;
    renderReportsList(reports);
  } catch (err) {
    showToast(err.message);
  }
}

function renderReportsList(reports) {
  clearEl(reportsListContainer);

  if (!reports || reports.length === 0) {
    const empty = createEl("p", {
      className: "empty-state",
      textContent: "No reports uploaded yet for this profile. Upload a report above to get started."
    });
    reportsListContainer.appendChild(empty);
    return;
  }

  reports.forEach(rep => {
    const dateText = formatDateString(rep.report_date);
    const labText = rep.lab_name ? rep.lab_name : "Lab not specified";
    const countText = `${rep.result_count} results`;

    const item = createEl("div", { className: "report-item", role: "listitem" });

    const mainClickable = createEl("div", {
      className: "report-item-main",
      tabindex: "0",
      role: "button",
      "aria-label": `Open report from ${dateText}`
    }, [
      createEl("span", { className: "report-item-title", textContent: `${dateText} - ${labText}` }),
      createEl("span", { className: "report-item-meta", textContent: countText })
    ]);

    mainClickable.addEventListener("click", () => loadReport(rep.report_id));
    mainClickable.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        loadReport(rep.report_id);
      }
    });

    const actions = createEl("div", { className: "report-item-actions" });
    const deleteBtn = createEl("button", {
      type: "button",
      className: "btn btn-danger-outline btn-sm",
      textContent: "Delete",
      "aria-label": `Delete report from ${dateText}`
    });

    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDeleteReport(rep.report_id, dateText);
    });

    actions.appendChild(deleteBtn);
    item.appendChild(mainClickable);
    item.appendChild(actions);

    reportsListContainer.appendChild(item);
  });
}

async function handleDeleteReport(reportId, dateLabel) {
  const confirmed = window.confirm(`Delete report from ${dateLabel}?`);
  if (!confirmed) return;

  try {
    await apiRequest(`/api/reports/${reportId}`, { method: "DELETE" });
    showToast("Report deleted.");
    if (state.currentReport && state.currentReport.report_id === reportId) {
      hideReportView();
    }
    loadReportsForProfile(state.activeProfileId);
  } catch (err) {
    showToast(err.message);
  }
}

function hideReportView() {
  state.currentReport = null;
  state.currentExplanation = null;
  state.currentQuestions = null;
  reportViewContainer.classList.add("hidden");
}

// Load and Display Full Report
async function loadReport(reportId, preloadedUpload = null) {
  try {
    announceLive("Loading report details...");

    const [report, explanation, questions, latest] = await Promise.all([
      preloadedUpload ? Promise.resolve(preloadedUpload) : apiRequest(`/api/reports/${reportId}`),
      apiRequest(`/api/reports/${reportId}/explanation?lang=${sessionExplanationLang}`),
      apiRequest(`/api/reports/${reportId}/questions`),
      apiRequest(`/api/profiles/${state.activeProfileId}/latest`)
    ]);

    state.currentReport = report;
    state.currentExplanation = explanation;
    state.currentQuestions = questions;
    state.latestTests = latest || {};

    renderReportView(report, explanation, questions);
    populateTrendDropdown();

    // Focus report heading
    reportMainHeading.setAttribute("tabindex", "-1");
    reportMainHeading.focus();
    announceLive("Report loaded.");
  } catch (err) {
    showToast(err.message);
  }
}

function renderReportView(report, explanation, questions) {
  reportViewContainer.classList.remove("hidden");

  // Header values
  reportDateValue.textContent = formatDateString(report.report_date);
  reportLabValue.textContent = report.lab_name || "Lab not specified";

  const results = report.results || [];
  reportTotalCount.textContent = `${results.length} total`;

  const abnormalCount = results.filter(r =>
    ["HIGH", "LOW", "CRITICAL_HIGH", "CRITICAL_LOW"].includes(r.flag)
  ).length;
  reportAbnormalCount.textContent = `${abnormalCount} abnormal`;

  // Extraction Method Badge
  let isOcr = false;
  if (report.extraction_method) {
    isOcr = report.extraction_method === "ocr";
  } else {
    isOcr = results.some(r => r.confidence != null && r.confidence < 1.0);
  }

  if (isOcr) {
    reportExtractionBadge.textContent = "Read from photo (OCR). Please check every value against your original report.";
    reportExtractionBadge.className = "badge badge-warning";
  } else {
    reportExtractionBadge.textContent = "Read from document text";
    reportExtractionBadge.className = "badge badge-info";
  }

  // Critical Escalation Banner
  const hasCritical = results.some(r => r.flag === "CRITICAL_LOW" || r.flag === "CRITICAL_HIGH");
  if (hasCritical) {
    criticalAlertBanner.classList.remove("hidden");
    const escalationText = (explanation && explanation.overall_notice)
      ? explanation.overall_notice
      : "One or more test results are far outside the usual range and need prompt medical attention. Please contact your doctor or a healthcare provider promptly.";
    criticalBannerMessage.textContent = escalationText;
  } else {
    criticalAlertBanner.classList.add("hidden");
  }

  // Warnings
  clearEl(reportWarningsList);
  const warnings = report.warnings || [];
  if (warnings.length > 0) {
    reportWarningsContainer.classList.remove("hidden");
    warnings.forEach(w => {
      reportWarningsList.appendChild(createEl("li", { textContent: w }));
    });
  } else {
    reportWarningsContainer.classList.add("hidden");
  }

  // Results Table
  clearEl(resultsTableBody);
  results.forEach((res, idx) => {
    const row = createEl("tr");

    // Test name column
    const testCell = createEl("td");
    const testWrapper = createEl("div", { className: "test-name-cell" });

    const canonName = res.test_name_canonical || res.test_name_raw;
    testWrapper.appendChild(createEl("span", { className: "canonical-name", textContent: canonName }));

    if (res.test_name_canonical && res.test_name_raw && res.test_name_canonical !== res.test_name_raw) {
      testWrapper.appendChild(createEl("span", { className: "raw-name", textContent: `(${res.test_name_raw})` }));
    }

    if (res.confidence != null && res.confidence < 0.85) {
      testWrapper.appendChild(createEl("span", {
        className: "verify-badge",
        textContent: VERIFY_TEXT
      }));
    }

    // Source line toggle button
    const snippetRowId = `snippet-row-${idx}`;
    const toggleSnippetBtn = createEl("button", {
      type: "button",
      className: "toggle-snippet-btn",
      textContent: "Show source line",
      "aria-expanded": "false",
      "aria-controls": snippetRowId
    });

    testWrapper.appendChild(toggleSnippetBtn);
    testCell.appendChild(testWrapper);

    // Your result column
    let resultDisplay = "";
    if (res.value != null) {
      resultDisplay = formatNumericValue(res.value);
      if (res.unit) resultDisplay += ` ${res.unit}`;
    } else if (res.value_text) {
      resultDisplay = res.value_text;
    } else {
      resultDisplay = "-";
    }
    const resultCell = createEl("td", { textContent: resultDisplay });

    // Reference range column
    let refDisplay = "-";
    if (res.ref_text) {
      refDisplay = res.ref_text;
    } else if (res.ref_low != null && res.ref_high != null) {
      refDisplay = `${res.ref_low} - ${res.ref_high}`;
    } else if (res.ref_high != null) {
      refDisplay = `< ${res.ref_high}`;
    } else if (res.ref_low != null) {
      refDisplay = `> ${res.ref_low}`;
    }
    const refCell = createEl("td", { textContent: refDisplay });

    // Status column
    const statusCfg = STATUS_CONFIG[res.flag] || STATUS_CONFIG.UNKNOWN;
    const statusCell = createEl("td", { className: "status-cell" }, [
      createEl("span", {
        className: `status-label ${statusCfg.className}`,
        textContent: statusCfg.label
      })
    ]);

    row.appendChild(testCell);
    row.appendChild(resultCell);
    row.appendChild(refCell);
    row.appendChild(statusCell);
    resultsTableBody.appendChild(row);

    // Collapsible snippet row
    const snippetRow = createEl("tr", { id: snippetRowId, className: "snippet-row hidden" });
    const snippetCell = createEl("td", { colspan: "4" });
    const snippetBox = createEl("div", {
      className: "snippet-content",
      textContent: res.source_snippet
        ? (res.source_page != null ? `[Page ${res.source_page}] ${res.source_snippet}` : res.source_snippet)
        : "No source line available."
    });
    snippetCell.appendChild(snippetBox);
    snippetRow.appendChild(snippetCell);
    resultsTableBody.appendChild(snippetRow);

    // Toggle event
    toggleSnippetBtn.addEventListener("click", () => {
      const isHidden = snippetRow.classList.contains("hidden");
      if (isHidden) {
        snippetRow.classList.remove("hidden");
        toggleSnippetBtn.textContent = "Hide source line";
        toggleSnippetBtn.setAttribute("aria-expanded", "true");
      } else {
        snippetRow.classList.add("hidden");
        toggleSnippetBtn.textContent = "Show source line";
        toggleSnippetBtn.setAttribute("aria-expanded", "false");
      }
    });
  });

  // Explanation Section
  if (explanationLangSelect) {
    explanationLangSelect.value = sessionExplanationLang;
  }
  renderExplanation(explanation);

  // Doctor Questions Section
  if (questions) {
    clearEl(doctorQuestionsOrdered);
    const qList = questions.questions || [];
    qList.forEach(q => {
      doctorQuestionsOrdered.appendChild(createEl("li", { textContent: q.text }));
    });

    clearEl(doctorTrendNotesList);
    const tNotes = questions.trend_notes || [];
    if (tNotes.length > 0) {
      trendNotesWrapper.classList.remove("hidden");
      tNotes.forEach(note => {
        doctorTrendNotesList.appendChild(createEl("li", { textContent: note }));
      });
    } else {
      trendNotesWrapper.classList.add("hidden");
    }

    clearEl(doctorGeneralQuestionsList);
    const gList = questions.general_questions || [];
    if (gList.length > 0) {
      generalQuestionsWrapper.classList.remove("hidden");
      gList.forEach(g => {
        doctorGeneralQuestionsList.appendChild(createEl("li", { textContent: g }));
      });
    } else {
      generalQuestionsWrapper.classList.add("hidden");
    }
  }
}

function renderExplanation(explanation) {
  if (!explanation) return;

  const lang = explanation.language || sessionExplanationLang || "en";
  if (explanationSection) {
    explanationSection.setAttribute("lang", lang);
  }

  if (explanationDraftNotice) {
    if (explanation.draft_notice) {
      explanationDraftNotice.textContent = explanation.draft_notice;
      explanationDraftNotice.classList.remove("hidden");
    } else {
      explanationDraftNotice.textContent = "";
      explanationDraftNotice.classList.add("hidden");
    }
  }

  if (explanationNotice) {
    explanationNotice.textContent = explanation.overall_notice || "";
  }

  if (explanationItemsList) {
    clearEl(explanationItemsList);
    const testExplanations = explanation.explanations || [];
    testExplanations.forEach(exp => {
      const card = createEl("div", { className: "explanation-card" });
      card.appendChild(createEl("div", { className: "explanation-card-header", textContent: exp.test_name }));
      card.appendChild(createEl("p", { className: "explanation-sentence", textContent: exp.result_sentence }));
      if (exp.what_it_measures) {
        card.appendChild(createEl("p", { className: "explanation-measures", textContent: exp.what_it_measures }));
      }
      if (exp.escalation) {
        card.appendChild(createEl("p", { className: "explanation-escalation", textContent: exp.escalation }));
      }
      explanationItemsList.appendChild(card);
    });
  }

  if (explanationDisclaimer) {
    explanationDisclaimer.textContent = explanation.disclaimer || "";
  }
}

// Copy Doctor Questions to Clipboard
function copyDoctorQuestions() {
  const q = state.currentQuestions;
  if (!q) return;

  const lines = ["Questions for your doctor:"];
  (q.questions || []).forEach((item, idx) => {
    lines.push(`${idx + 1}. ${item.text}`);
  });

  if (q.trend_notes && q.trend_notes.length > 0) {
    lines.push("\nTrend observations:");
    q.trend_notes.forEach(note => lines.push(`- ${note}`));
  }

  if (q.general_questions && q.general_questions.length > 0) {
    lines.push("\nGeneral questions:");
    q.general_questions.forEach(item => lines.push(`- ${item}`));
  }

  const textToCopy = lines.join("\n");

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(textToCopy).then(() => {
      copyStatusMsg.textContent = "Copied questions to clipboard!";
      setTimeout(() => { copyStatusMsg.textContent = ""; }, 3000);
      showToast("Questions copied to clipboard.");
    }).catch(() => {
      fallbackCopyText(textToCopy);
    });
  } else {
    fallbackCopyText(textToCopy);
  }
}

function fallbackCopyText(text) {
  copyStatusMsg.textContent = "Could not copy automatically. Please select and copy manually.";
  setTimeout(() => { copyStatusMsg.textContent = ""; }, 4000);
}

// Trends Section
function populateTrendDropdown() {
  clearEl(trendTestSelect);

  const tests = Object.keys(state.latestTests || {}).sort();
  if (tests.length === 0) {
    const opt = createEl("option", { value: "", textContent: "No numeric tests available" });
    trendTestSelect.appendChild(opt);
    trendTestSelect.disabled = true;
    renderEmptyTrend();
    return;
  }

  trendTestSelect.disabled = false;
  tests.forEach(testName => {
    const opt = createEl("option", { value: testName, textContent: testName });
    trendTestSelect.appendChild(opt);
  });

  // Try to default to HbA1c or the first test from the current report
  let defaultTest = tests[0];
  const reportCanonicalNames = (state.currentReport && state.currentReport.results)
    ? state.currentReport.results.map(r => r.test_name_canonical).filter(Boolean)
    : [];

  if (tests.includes("HbA1c")) {
    defaultTest = "HbA1c";
  } else {
    for (const cn of reportCanonicalNames) {
      if (tests.includes(cn)) {
        defaultTest = cn;
        break;
      }
    }
  }

  trendTestSelect.value = defaultTest;
  state.selectedTrendTest = defaultTest;
  renderTrendForSelectedTest();
}

function renderEmptyTrend() {
  trendSinglePointNote.classList.add("hidden");
  clearEl(trendDataTableBody);
  clearEl(trendPlot);
}

async function renderTrendForSelectedTest() {
  const testName = state.selectedTrendTest;
  if (!testName) return;

  try {
    const points = await apiRequest(`/api/profiles/${state.activeProfileId}/trends/${encodeURIComponent(testName)}`);
    state.trendData = points || [];
    await renderPlotlyChartAndTable(testName, state.trendData);
  } catch (err) {
    showToast(err.message);
  }
}

// Compute date range +/- N days in UTC arithmetic to prevent timezone shift
function computeUtcDateRange(dateStr, days) {
  const parts = dateStr.split("-").map(n => parseInt(n, 10));
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  const offsetMs = (days || 15) * 86400000;
  const startDt = new Date(dt.getTime() - offsetMs);
  const endDt = new Date(dt.getTime() + offsetMs);

  function toUtcYmd(d) {
    const y = d.getUTCFullYear();
    const m = String(d.getUTCMonth() + 1).padStart(2, "0");
    const dayStr = String(d.getUTCDate()).padStart(2, "0");
    return `${y}-${m}-${dayStr}`;
  }

  return [toUtcYmd(startDt), toUtcYmd(endDt)];
}

async function renderPlotlyChartAndTable(testName, points) {
  clearEl(trendDataTableBody);

  if (!points || points.length === 0) {
    trendSinglePointNote.classList.add("hidden");
    clearEl(trendPlot);
    return;
  }

  // Populate plain data table alternative
  points.forEach(pt => {
    const row = createEl("tr");
    row.appendChild(createEl("td", { textContent: pt.report_date }));
    row.appendChild(createEl("td", { textContent: formatNumericValue(pt.value) }));
    row.appendChild(createEl("td", { textContent: pt.unit || "-" }));

    let refStr = "-";
    if (pt.ref_low != null && pt.ref_high != null) {
      refStr = `${pt.ref_low} - ${pt.ref_high}`;
    } else if (pt.ref_high != null) {
      refStr = `< ${pt.ref_high}`;
    } else if (pt.ref_low != null) {
      refStr = `> ${pt.ref_low}`;
    }
    row.appendChild(createEl("td", { textContent: refStr }));

    const statusCfg = STATUS_CONFIG[pt.flag] || STATUS_CONFIG.UNKNOWN;
    const statusCell = createEl("td", { className: "status-cell" }, [
      createEl("span", {
        className: `status-label ${statusCfg.className}`,
        textContent: statusCfg.label
      })
    ]);
    row.appendChild(statusCell);

    trendDataTableBody.appendChild(row);
  });

  // Handle single data point note
  if (points.length === 1) {
    trendSinglePointNote.textContent = "Upload another report to see a trend.";
    trendSinglePointNote.classList.remove("hidden");
  } else {
    trendSinglePointNote.classList.add("hidden");
  }

  // Load Plotly vendor script dynamically on first need
  const Plotly = await loadPlotly();

  const dates = points.map(p => p.report_date);
  const values = points.map(p => p.value);
  const symbols = points.map(p => getPlotlyShapeForFlag(p.flag));
  const unit = points[0] ? points[0].unit : "";

  const trace = {
    x: dates,
    y: values,
    mode: "lines+markers",
    type: "scatter",
    line: { color: "#0284c7", width: 2 },
    marker: {
      symbol: symbols,
      size: 10,
      color: "#0284c7",
      line: { color: "#0f172a", width: 1 }
    },
    hoverinfo: "x+y"
  };

  // Determine shaded reference band using ref_low and ref_high
  const shapes = [];
  const latestPt = points[points.length - 1];
  const refLow = latestPt.ref_low;
  const refHigh = latestPt.ref_high;

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  let bandLow = null;
  let bandHigh = null;

  if (refLow != null && refHigh != null) {
    bandLow = refLow;
    bandHigh = refHigh;
  } else if (refHigh != null) {
    // One-sided "< limit": shade from axis minimum up to the limit
    const axisMin = Math.min(0, minVal * 0.8);
    bandLow = axisMin;
    bandHigh = refHigh;
  } else if (refLow != null) {
    // One-sided "> limit": shade from limit up to axis maximum
    const axisMax = Math.max(maxVal * 1.2, refLow * 1.5);
    bandLow = refLow;
    bandHigh = axisMax;
  }

  if (bandLow != null && bandHigh != null) {
    shapes.push({
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: bandLow,
      y1: bandHigh,
      fillcolor: "rgba(34, 197, 94, 0.15)",
      line: { width: 0 },
      layer: "below"
    });
  }

  const xaxisConfig = {
    title: { text: "Date" },
    type: "date",
    tickformat: "%b %d, %Y",
    automargin: true
  };

  // For a single data point, set range to +/- 15 days in UTC with at most 3 ticks
  if (points.length === 1) {
    const [rangeStart, rangeEnd] = computeUtcDateRange(points[0].report_date, 15);
    xaxisConfig.range = [rangeStart, rangeEnd];
    xaxisConfig.nticks = 3;
  }

  const layout = {
    margin: { t: 25, r: 25, b: 50, l: 60 },
    xaxis: xaxisConfig,
    yaxis: {
      title: { text: unit || "" },
      automargin: true
    },
    showlegend: false,
    shapes: shapes,
    autosize: true
  };

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot(trendPlot, [trace], layout, config);
}

// Prescription Summary Render
function renderPrescriptionSummary(summary) {
  prescriptionViewContainer.classList.remove("hidden");
  prescriptionHeading.setAttribute("tabindex", "-1");
  prescriptionHeading.focus();

  // Medications list
  clearEl(medicationsCardList);
  const meds = summary.medications || [];

  if (meds.length === 0) {
    medicationsCardList.appendChild(createEl("p", {
      className: "empty-state",
      textContent: "No medications could be extracted from this prescription."
    }));
  } else {
    meds.forEach((med, idx) => {
      const card = createEl("div", { className: "medication-card" });

      const header = createEl("div", { className: "med-header" }, [
        createEl("span", { className: "med-name", textContent: med.name_as_written })
      ]);
      card.appendChild(header);

      const grid = createEl("div", { className: "med-details-grid" });

      grid.appendChild(createMedField("Form", med.form || "-"));
      grid.appendChild(createMedField("Strength", med.strength || "-"));
      grid.appendChild(createMedField("Frequency", formatWithWords(med.frequency_raw, med.frequency_plain)));
      grid.appendChild(createMedField("Duration", med.duration_raw || "-"));
      grid.appendChild(createMedField("Instructions", formatWithWords(med.instructions_raw, med.instructions_plain)));

      card.appendChild(grid);

      // Source snippet toggle
      const snippetId = `med-snippet-${idx}`;
      const toggleSnippetBtn = createEl("button", {
        type: "button",
        className: "toggle-snippet-btn",
        textContent: "Show source line",
        "aria-expanded": "false",
        "aria-controls": snippetId
      });

      const snippetBox = createEl("div", {
        id: snippetId,
        className: "med-snippet snippet-content hidden",
        textContent: med.source_snippet || "No source snippet available."
      });

      toggleSnippetBtn.addEventListener("click", () => {
        const isHidden = snippetBox.classList.contains("hidden");
        if (isHidden) {
          snippetBox.classList.remove("hidden");
          toggleSnippetBtn.textContent = "Hide source line";
          toggleSnippetBtn.setAttribute("aria-expanded", "true");
        } else {
          snippetBox.classList.add("hidden");
          toggleSnippetBtn.textContent = "Show source line";
          toggleSnippetBtn.setAttribute("aria-expanded", "false");
        }
      });

      card.appendChild(toggleSnippetBtn);
      card.appendChild(snippetBox);

      medicationsCardList.appendChild(card);
    });
  }

  // Other lines
  clearEl(prescriptionOtherLinesList);
  const otherLines = summary.other_lines || [];
  if (otherLines.length > 0) {
    prescriptionOtherLinesSection.classList.remove("hidden");
    otherLines.forEach(line => {
      prescriptionOtherLinesList.appendChild(createEl("li", { textContent: line }));
    });
  } else {
    prescriptionOtherLinesSection.classList.add("hidden");
  }

  // Warnings
  clearEl(prescriptionWarningsList);
  const warnings = summary.warnings || [];
  if (warnings.length > 0) {
    prescriptionWarningsSection.classList.remove("hidden");
    warnings.forEach(warn => {
      prescriptionWarningsList.appendChild(createEl("li", { textContent: warn }));
    });
  } else {
    prescriptionWarningsSection.classList.add("hidden");
  }

  // Disclaimer
  prescriptionDisclaimerText.textContent = summary.disclaimer || "";
}

function createMedField(label, value) {
  return createEl("div", { className: "med-field" }, [
    createEl("span", { className: "med-field-label", textContent: label }),
    createEl("span", { className: "med-field-value", textContent: value })
  ]);
}
