const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Multimodal Parkinson\'s Disease Detection';
pres.author = 'VIT-AP University';

// ===== THEME COLORS =====
const C = {
  navy:    "0B2A6F",  // deep blue (primary)
  blue:    "1D4ED8",  // medium blue
  teal:    "0D9488",  // teal accent
  tealLt:  "CCFBF1",  // light teal bg
  blueLt:  "DBEAFE",  // light blue bg
  white:   "FFFFFF",
  dark:    "111827",
  gray:    "6B7280",
  grayLt:  "F3F4F6",
  grayMd:  "E5E7EB",
  crimson: "8B1A1A",  // VIT-AP red
  text:    "1F2937",
  textMd:  "374151",
  textSm:  "6B7280",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.10 });

// Helper: left accent bar (consistent with HTML slides)
function addAccentBar(slide) {
  // Navy top bar
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.07, fill: { color: C.navy }, line: { color: C.navy } });
  // Left rail
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.07, w: 0.2, h: 3.5, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.57, w: 0.2, h: 1.2, fill: { color: C.blue }, line: { color: C.blue } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.77, w: 0.2, h: 0.855, fill: { color: C.teal }, line: { color: C.teal } });
}

// Helper: section tag chip
function addTag(slide, text, x, y, bgColor, textColor) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: text.length * 0.085 + 0.25, h: 0.25, fill: { color: bgColor }, line: { color: bgColor }, rectRadius: 0.04 });
  slide.addText(text, { x, y, w: text.length * 0.085 + 0.25, h: 0.25, fontSize: 8, color: textColor, bold: true, align: "center", valign: "middle" });
}

// Helper: stat card
function addStatCard(slide, x, y, w, h, label, value, bgColor, textColor) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bgColor }, line: { color: bgColor }, shadow: makeShadow() });
  slide.addText(label, { x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.22, fontSize: 8.5, color: textColor, bold: true });
  slide.addText(value, { x: x + 0.12, y: y + 0.28, w: w - 0.24, h: 0.38, fontSize: 24, color: textColor, bold: true, align: "left" });
}

// Helper: section header with icon box
// iconLabel = 2-3 char abbreviation shown inside the badge
function addSlideHeader(slide, moduleLabel, title, titleSize = 30, iconLabel = null, iconColor = null) {
  addAccentBar(slide);
  const ic = iconLabel || moduleLabel.substring(0, 2).toUpperCase();
  const bg = iconColor || C.navy;
  // Module label (small caps above icon)
  slide.addText(moduleLabel.toUpperCase(), { x: 0.32, y: 0.1, w: 3, h: 0.18, fontSize: 7.5, color: C.blue, bold: true, charSpacing: 3 });
  // Icon badge with abbreviation
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 0.28, w: 0.42, h: 0.38, fill: { color: bg }, line: { color: bg } });
  slide.addText(ic, { x: 0.32, y: 0.28, w: 0.42, h: 0.38, fontSize: ic.length > 2 ? 9 : 11, color: C.white, bold: true, align: "center", valign: "middle" });
  // Title
  slide.addText(title, { x: 0.86, y: 0.22, w: 7.0, h: 0.48, fontSize: titleSize, color: C.dark, bold: true, fontFace: "Calibri" });
}

// Helper: content card
function addCard(slide, x, y, w, h, bgColor = C.white, borderColor = C.grayMd) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bgColor }, line: { color: borderColor, width: 1 }, shadow: makeShadow() });
}

// =============================================
// SLIDE 1 — TITLE SLIDE (REDESIGNED - NO STAT CARDS)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "F0F4FF" };

  // Full-width navy header band
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.55, fill: { color: C.navy }, line: { color: C.navy } });
  // Teal accent stripe
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.55, w: 10, h: 0.06, fill: { color: C.teal }, line: { color: C.teal } });
  // Bottom footer band
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.32, w: 10, h: 0.06, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.38, w: 10, h: 0.245, fill: { color: C.navy }, line: { color: C.navy } });

  // Left teal accent bars in header
  s.addShape(pres.shapes.RECTANGLE, { x: 8.6, y: 0.12, w: 0.1, h: 1.1, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.85, y: 0.32, w: 0.1, h: 0.72, fill: { color: C.blue }, line: { color: C.blue } });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.1, y: 0.52, w: 0.1, h: 0.44, fill: { color: "4ADE80" }, line: { color: "4ADE80" } });

  // University info
  s.addText("VIT-AP UNIVERSITY", { x: 0.42, y: 0.1, w: 7.5, h: 0.38, fontSize: 22, color: C.white, bold: true, fontFace: "Calibri" });
  s.addText("School of Computer Science and Engineering", { x: 0.42, y: 0.5, w: 8, h: 0.22, fontSize: 12, color: "A5B4FC", fontFace: "Calibri" });
  s.addText("Senior Design Project Review - 2  |  Dept. of CSE  |  Amaravati, Andhra Pradesh", { x: 0.42, y: 0.74, w: 8, h: 0.2, fontSize: 10.5, color: "93C5FD", fontFace: "Calibri" });
  s.addText("Academic Year 2024-25", { x: 0.42, y: 0.97, w: 4, h: 0.18, fontSize: 9.5, color: "7DD3FC", fontFace: "Calibri" });

  // Main title area - large, centered
  s.addText("Multimodal Parkinson's Disease Detection", { x: 0.4, y: 1.72, w: 9.2, h: 0.72, fontSize: 34, color: C.navy, bold: true, fontFace: "Calibri", align: "center" });

  // Decorative horizontal rule
  s.addShape(pres.shapes.RECTANGLE, { x: 3.2, y: 2.48, w: 3.6, h: 0.04, fill: { color: C.teal }, line: { color: C.teal } });

  s.addText("Using Cross-Modal Attention Fusion of", { x: 0.4, y: 2.56, w: 9.2, h: 0.4, fontSize: 20, color: C.blue, fontFace: "Calibri", align: "center" });
  s.addText("Handwriting and Speech Biomarkers", { x: 0.4, y: 2.93, w: 9.2, h: 0.36, fontSize: 20, color: C.teal, fontFace: "Calibri", align: "center", bold: true });

  // Three feature pills
  const pills = [
    ["Handwriting Biomarkers", "Spiral / Wave + 16 spatial features", C.navy],
    ["Speech Biomarkers", "XLS-R + MFCC + Praat acoustics", C.teal],
    ["CMAFN Fusion", "Cross-attention + GMU + Web App", C.blue],
  ];
  pills.forEach(([title, sub, col], i) => {
    const x = 0.35 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.42, w: 3.0, h: 0.72, fill: { color: col }, line: { color: col }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.18 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.42, w: 3.0, h: 0.04, fill: { color: "4ADE80" }, line: { color: "4ADE80" } });
    s.addText(title, { x: x + 0.1, y: 3.46, w: 2.8, h: 0.28, fontSize: 11.5, color: C.white, bold: true, align: "center", valign: "middle" });
    s.addText(sub, { x: x + 0.1, y: 3.74, w: 2.8, h: 0.36, fontSize: 8.5, color: "CCFBF1", align: "center", valign: "middle" });
  });

  // Authors section — two side-by-side cards
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.28, w: 4.3, h: 0.92, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.09 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.28, w: 0.06, h: 0.92, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("Guide", { x: 0.5, y: 4.32, w: 4.05, h: 0.2, fontSize: 8.5, color: C.gray, bold: true, charSpacing: 1 });
  s.addText("Dr. Rajasekhar Boddu", { x: 0.5, y: 4.52, w: 4.05, h: 0.24, fontSize: 13, color: C.navy, bold: true });
  s.addText("Assistant Professor, Dept. of CSE, VIT-AP University", { x: 0.5, y: 4.76, w: 4.05, h: 0.2, fontSize: 9, color: C.gray });

  s.addShape(pres.shapes.RECTANGLE, { x: 4.95, y: 4.28, w: 4.7, h: 0.92, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.09 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.95, y: 4.28, w: 0.06, h: 0.92, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("Presented by", { x: 5.1, y: 4.32, w: 4.45, h: 0.2, fontSize: 8.5, color: C.gray, bold: true, charSpacing: 1 });
  s.addText("T. V. Thanuj (22BCE20003)   K. K. Chaitanya (22BCE7359)", { x: 5.1, y: 4.5, w: 4.45, h: 0.22, fontSize: 10, color: C.navy, bold: true });
  s.addText("P. Narendar Reddy (22BCE7707)", { x: 5.1, y: 4.72, w: 4.45, h: 0.2, fontSize: 10, color: C.navy, bold: true });

  s.addText("VIT-AP  |  CSE  |  Multimodal Parkinson's Disease Detection using Cross-Modal Attention Fusion", { x: 0.3, y: 5.4, w: 9.4, h: 0.2, fontSize: 7.5, color: C.white, align: "center" });
}

// =============================================
// SLIDE 2 — OUTLINE
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addAccentBar(s);

  addSlideHeader(s, "Contents", "Presentation Outline", 30, "OL");

  const items = [
    ["01", "Abstract & Introduction", "Problem, motivation, and approach"],
    ["02", "Literature Review", "25 existing systems (handwriting, speech, multimodal)"],
    ["03", "System Architecture", "Three-module end-to-end pipeline"],
    ["04", "Methodology", "Module A: Handwriting • Module B: Speech • Module C: CMAFN"],
    ["05", "Datasets & Experimental Setup", "3,264 images • 831 speech recordings • Patient-level CV"],
    ["06", "Results & Performance", "Accuracy • AUC-ROC • Fold-level stability"],
    ["07", "Key Contributions & Web App", "CMAFN fusion • Flask deployment • <3s inference"],
    ["08", "Conclusion & Future Work", "Multimodal gain • Roadmap"],
  ];

  items.forEach(([num, title, sub], i) => {
    const col = i < 4 ? 0 : 1;
    const row = i % 4;
    const x = 0.32 + col * 4.85;
    const y = 1.05 + row * 1.08;

    addCard(s, x, y, 4.55, 0.92, i % 2 === 0 ? C.grayLt : C.white);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.38, h: 0.92, fill: { color: i < 4 ? C.navy : C.teal }, line: { color: i < 4 ? C.navy : C.teal } });
    s.addText(num, { x, y: y + 0.3, w: 0.38, h: 0.3, fontSize: 12, color: C.white, bold: true, align: "center" });
    s.addText(title, { x: x + 0.48, y: y + 0.12, w: 3.95, h: 0.28, fontSize: 12, color: C.navy, bold: true });
    s.addText(sub, { x: x + 0.48, y: y + 0.4, w: 3.95, h: 0.38, fontSize: 9, color: C.gray });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 3 — ABSTRACT
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Overview", "Abstract", 30, "AB");

  addCard(s, 0.32, 1.0, 9.38, 3.65);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 1.0, w: 9.38, h: 0.07, fill: { color: C.navy }, line: { color: C.navy } });

  s.addText(
    "Parkinson's disease (PD) is a progressive neurodegenerative disorder where clinical diagnosis often occurs late, after significant dopaminergic neuron loss. This work proposes a multimodal deep learning framework for non-invasive PD screening by fusing handwriting and speech biomarkers through a novel Cross-Modal Attention Fusion Network (CMAFN). Module A analyzes spiral/wave drawings using 16 engineered spatial biomarkers combined with an EfficientNet-B0 CNN via a meta-learner ensemble, achieving ~93.6% accuracy. Module B processes short voice samples through four complementary pathways — XLS-R self-supervised embeddings, Mel-spectrogram CNN, MFCC, and Praat acoustic measures — fused by multi-head cross-attention, achieving ~91.7% accuracy. Module C (CMAFN) integrates both modalities via bidirectional Transformer cross-attention and a Gated Multimodal Unit (GMU), achieving 96.94% accuracy and 0.9995 AUC-ROC. All evaluation uses patient-level stratified 5-fold cross-validation to prevent data leakage. The system is deployed as a Flask web application supporting real-time inference in under 3 seconds on CPU, enabling practical, scalable, and non-invasive PD screening using consumer hardware.",
    { x: 0.5, y: 1.15, w: 9.05, h: 3.4, fontSize: 12.5, color: C.textMd, align: "justify", valign: "top", fontFace: "Calibri", paraSpaceAfter: 4 }
  );

  // Key stats strip
  const kstats = [["96.94%", "Fusion Accuracy"], ["0.9995", "AUC-ROC"], ["3 Modules", "End-to-end"], ["5-fold", "Patient-level CV"]];
  kstats.forEach(([v, l], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32 + i * 2.35, y: 4.75, w: 2.25, h: 0.58, fill: { color: i === 0 ? C.navy : i === 1 ? C.teal : C.grayLt }, line: { color: C.grayMd } });
    s.addText(v, { x: 0.32 + i * 2.35, y: 4.78, w: 2.25, h: 0.3, fontSize: 16, color: i < 2 ? C.white : C.navy, bold: true, align: "center" });
    s.addText(l, { x: 0.32 + i * 2.35, y: 5.06, w: 2.25, h: 0.2, fontSize: 8.5, color: i < 2 ? "A5B4FC" : C.gray, align: "center" });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 4 — PROBLEM STATEMENT & MOTIVATION
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Context", "Problem Statement & Motivation", 24, "PS");

  // Two column cards
  addCard(s, 0.32, 1.0, 4.55, 4.28);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 1.0, w: 4.55, h: 0.32, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("PROBLEM", { x: 0.42, y: 1.03, w: 4.35, h: 0.26, fontSize: 11, color: C.white, bold: true, charSpacing: 2 });

  const probLabels = ["01", "02", "03", "04"];
  const probs = [
    ["Late Diagnosis", "~60–80% of dopaminergic neurons are lost before motor symptoms become clinically observable."],
    ["High-Cost Tests", "Neuroimaging (SPECT/PET) and specialist assessments are expensive and not widely accessible."],
    ["Subjective Scales", "Clinical rating scales (UPDRS, H&Y) vary across examiners; early-stage sensitivity is limited."],
    ["Limited Access", "Remote and under-resourced settings lack specialized neurologist access for early PD screening."],
  ];
  probs.forEach(([h, body], i) => {
    const y = 1.42 + i * 0.95;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.42, y, w: 0.38, h: 0.75, fill: { color: C.blueLt }, line: { color: C.navy, width: 0.5 } });
    s.addText(probLabels[i], { x: 0.42, y: y + 0.22, w: 0.38, h: 0.28, fontSize: 12, color: C.navy, bold: true, align: "center" });
    s.addText(h, { x: 0.9, y: y + 0.06, w: 3.85, h: 0.24, fontSize: 11.5, color: C.navy, bold: true });
    s.addText(body, { x: 0.9, y: y + 0.34, w: 3.85, h: 0.36, fontSize: 9.5, color: C.textMd });
  });

  addCard(s, 5.05, 1.0, 4.65, 4.28);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.05, y: 1.0, w: 4.65, h: 0.32, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("PROPOSED SOLUTION", { x: 5.15, y: 1.03, w: 4.45, h: 0.26, fontSize: 11, color: C.white, bold: true, charSpacing: 2 });

  const solLabels = ["HW", "SP", "AI", "WB"];
  const sols = [
    ["Handwriting Biomarkers", "Spiral/wave drawings + 16 spatial features quantify fine motor degradation (micrographia, tremor cues)."],
    ["Speech Biomarkers", "XLS-R embeddings + MFCC + Praat measures capture dysarthria, jitter, shimmer, and hypophonia patterns."],
    ["Cross-Modal Fusion", "CMAFN bidirectional Transformer cross-attention + Gated Multimodal Unit (GMU) learns complementary interactions."],
    ["Deployable Web App", "Flask-based real-time screening; <3s CPU inference; non-invasive; consumer hardware compatible."],
  ];
  sols.forEach(([h, body], i) => {
    const y = 1.42 + i * 0.95;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y, w: 0.38, h: 0.75, fill: { color: C.tealLt }, line: { color: C.teal, width: 0.5 } });
    s.addText(solLabels[i], { x: 5.15, y: y + 0.22, w: 0.38, h: 0.28, fontSize: 10, color: C.teal, bold: true, align: "center" });
    s.addText(h, { x: 5.63, y: y + 0.06, w: 3.95, h: 0.24, fontSize: 11.5, color: C.teal, bold: true });
    s.addText(body, { x: 5.63, y: y + 0.34, w: 3.95, h: 0.36, fontSize: 9.5, color: C.textMd });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 5 — OBJECTIVES
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Plan", "Research Objectives", 30, "OBJ");

  const objs = [
    ["01", "Handwriting Analyzer", "Train a dual-pathway model combining 16 engineered spatial biomarkers (MLP) with raw image learning (EfficientNet-B0 + CBAM attention). Meta-learner: Logistic Regression stacking.", C.navy, C.blueLt],
    ["02", "Multi-Pathway Speech Engine", "Combine self-supervised XLS-R representations (BiLSTM) with Mel-spectrogram CNN, MFCC (+Δ), and Praat acoustic voice-quality measures. Fused via multi-head cross-attention.", C.teal, C.tealLt],
    ["03", "16 Spatial Biomarkers", "Extract clinically relevant handwriting features: stroke width mean/std, curvature, direction changes, contour roughness, connected component density, ink density, entropy, fractal dimension, Hu moments.", C.navy, C.blueLt],
    ["04", "CMAFN Fusion Network", "Design Transformer-based bidirectional cross-attention (HW↔Speech) and a Gated Multimodal Unit (GMU) to learn complementary interactions across motor + voice biomarkers.", C.teal, C.tealLt],
    ["05", "Real-Time Web Deployment", "Deploy as a Flask web application with interactive handwriting and audio inputs; CPU-friendly inference (<3s); INT8 quantized model (~85 MB); patient-level validation.", C.navy, C.blueLt],
  ];

  objs.forEach(([num, title, body, accent, bg], i) => {
    const y = 1.02 + i * 0.87;
    addCard(s, 0.32, y, 9.38, 0.78, bg);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y, w: 0.5, h: 0.78, fill: { color: accent }, line: { color: accent } });
    s.addText(num, { x: 0.32, y: y + 0.25, w: 0.5, h: 0.28, fontSize: 14, color: C.white, bold: true, align: "center" });
    s.addText(title, { x: 0.94, y: y + 0.1, w: 2.4, h: 0.28, fontSize: 12, color: accent, bold: true });
    s.addText(body, { x: 0.94, y: y + 0.38, w: 8.55, h: 0.35, fontSize: 9.5, color: C.textMd });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 3.35, y: 1.06, w: 0.006, h: 4.25, fill: { color: C.grayMd }, line: { color: C.grayMd } });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 6 — LITERATURE: Handwriting (FULL PAGE BEAUTIFUL)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "0D1B3E" };

  // Teal left strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });
  // Top header band
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0, w: 9.78, h: 0.72, fill: { color: "0B2252" }, line: { color: "0B2252" } });
  // Teal bottom accent
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0.72, w: 9.78, h: 0.05, fill: { color: C.teal }, line: { color: C.teal } });

  // Header text
  s.addText("TABLE 1: LITERATURE SURVEY", { x: 7.0, y: 0.12, w: 2.9, h: 0.22, fontSize: 9, color: C.teal, bold: true, align: "right", charSpacing: 1 });
  s.addText("HW", { x: 0.32, y: 0.06, w: 0.68, h: 0.58, fontSize: 14, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("Existing Systems: Handwriting-Based PD Detection", { x: 1.1, y: 0.1, w: 8.6, h: 0.52, fontSize: 22, color: C.white, bold: true, fontFace: "Calibri" });

  // Table header row
  const hCols = [0.22, 0.78, 4.12, 6.48, 8.72];
  const hWidths = [0.56, 3.34, 2.36, 2.24, 1.06];
  const hLabels = ["Sl", "Title & Year", "Methodology", "Dataset", "Acc"];
  hCols.forEach((x, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.77, w: hWidths[i], h: 0.38, fill: { color: "0D3B6E" }, line: { color: "0D3B6E" } });
    s.addText(hLabels[i], { x: x+0.06, y: 0.77, w: hWidths[i]-0.08, h: 0.38, fontSize: 10.5, color: C.white, bold: true, align: i===4?"center":"left", valign: "middle" });
  });

  const hwRows = [
    ["1","Base Paper: Automated PD Detection using DL on Spiral Drawings (2023)","ResNet50 + Transfer Learning","NewHandPD (204 samples)","87.2%"],
    ["2","CNN-Based Handwriting Analysis for PD Diagnosis (2024)","EfficientNet-B3 + Data Augmentation","Custom (450 images)","89.5%"],
    ["3","Vision Transformer for Parkinson's Detection from Drawings (2023)","ViT-Base with Fine-tuning","PaHaW Dataset","85.7%"],
    ["4","Ensemble Deep Learning for PD Screening via Handwriting (2022)","ResNet + VGG + Inception Ensemble","Combined (600 samples)","88.8%"],
    ["5","GAN-Augmented Deep Learning for Micrographia Detection (2023)","StyleGAN2 + DenseNet121","Synthetic + Real (800 imgs)","86.4%"],
    ["6","Attention-Based CNN for Spiral Drawing Analysis (2024)","Custom CNN with Spatial Attention","NewHandPD + PaHaW","90.1%"],
    ["7","Multi-Task Learning for PD Severity Assessment (2022)","MTL-CNN (Classification + Regression)","Clinical Dataset (380 samples)","84.6%"],
    ["8","Explainable AI for Handwriting-Based PD Detection (2023)","ResNet34 + GradCAM Visualization","NewHandPD","87.8%"],
  ];

  hwRows.forEach((row, ri) => {
    const ry = 1.15 + ri * 0.545;
    const bg = ri % 2 === 0 ? "132242" : "172A4E";
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 9.78, h: 0.52, fill: { color: bg }, line: { color: bg } });
    // Sl badge
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 0.56, h: 0.52, fill: { color: ri % 2 === 0 ? "0B2252" : "112040" }, line: { color: "1E3A6A" } });
    s.addText(row[0], { x: 0.22, y: ry, w: 0.56, h: 0.52, fontSize: 11, color: "93C5FD", bold: true, align: "center", valign: "middle" });
    // Title
    s.addText(row[1], { x: 0.84, y: ry+0.04, w: 3.28, h: 0.44, fontSize: 9, color: C.white, valign: "middle" });
    // Methodology - highlighted
    s.addText(row[2], { x: 4.18, y: ry+0.04, w: 2.24, h: 0.44, fontSize: 9, color: "93C5FD", bold: true, valign: "middle" });
    // Dataset
    s.addText(row[3], { x: 6.48, y: ry+0.04, w: 2.18, h: 0.44, fontSize: 9, color: "CBD5E1", valign: "middle" });
    // Accuracy - green badge
    s.addShape(pres.shapes.RECTANGLE, { x: 8.72, y: ry+0.1, w: 1.06, h: 0.32, fill: { color: "052E16" }, line: { color: "166534" } });
    s.addText(row[4], { x: 8.72, y: ry+0.1, w: 1.06, h: 0.32, fontSize: 11, color: "4ADE80", bold: true, align: "center", valign: "middle" });
    // row separator
    s.addShape(pres.shapes.LINE, { x: 0.22, y: ry+0.52, w: 9.78, h: 0, line: { color: "1E3A6A", width: 0.5 } });
  });

  // Column separators
  [0.78, 4.12, 6.48, 8.72].forEach(x => {
    s.addShape(pres.shapes.LINE, { x, y: 0.77, w: 0, h: 4.51, line: { color: "1E3A6A", width: 0.5 } });
  });

  s.addText("Slide 6  •  Multi-Modal Parkinson's Disease Detection System", { x: 0.28, y: 5.47, w: 9.44, h: 0.15, fontSize: 8, color: "4A6FA5", align: "left" });
}

// =============================================
// SLIDE 7 — LITERATURE: Speech (FULL PAGE BEAUTIFUL)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "0D1B3E" };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: "0D9488" }, line: { color: "0D9488" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0, w: 9.78, h: 0.72, fill: { color: "062A2A" }, line: { color: "062A2A" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0.72, w: 9.78, h: 0.05, fill: { color: "0D9488" }, line: { color: "0D9488" } });

  s.addText("TABLE 2: LITERATURE SURVEY", { x: 7.0, y: 0.12, w: 2.9, h: 0.22, fontSize: 9, color: "0D9488", bold: true, align: "right", charSpacing: 1 });
  s.addText("SP", { x: 0.32, y: 0.06, w: 0.68, h: 0.58, fontSize: 14, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("Existing Systems: Speech-Based PD Detection", { x: 1.1, y: 0.1, w: 8.6, h: 0.52, fontSize: 22, color: C.white, bold: true, fontFace: "Calibri" });

  const hCols = [0.22, 0.78, 4.12, 6.48, 8.72];
  const hWidths = [0.56, 3.34, 2.36, 2.24, 1.06];
  const hLabels = ["Sl", "Title & Year", "Methodology", "Dataset", "Acc"];
  hCols.forEach((x, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.77, w: hWidths[i], h: 0.38, fill: { color: "073B3B" }, line: { color: "073B3B" } });
    s.addText(hLabels[i], { x: x+0.06, y: 0.77, w: hWidths[i]-0.08, h: 0.38, fontSize: 10.5, color: C.white, bold: true, align: i===4?"center":"left", valign: "middle" });
  });

  const spRows = [
    ["9","Deep Learning on Voice Recordings for PD Diagnosis (2024)","1D-CNN on MFCC Features","PC-GITA Dataset","83.5%"],
    ["10","Transformer-Based Speech Analysis for Parkinson's (2023)","Speech Transformer + Acoustic Features","Italian Parkinson's Voice","85.3%"],
    ["11","CNN-LSTM Hybrid for PD Voice Analysis (2022)","BiLSTM + CNN on Spectrograms","mPower Dataset (1,200)","84.2%"],
    ["12","Multi-Feature Fusion for PD Speech Detection (2024)","XGBoost on 132 Acoustic Features","Sakar Dataset","86.7%"],
    ["13","Mel-Spectrogram CNN for Dysarthria Detection (2023)","ResNet18 on Mel-Spectrograms","Custom (650 recordings)","83.9%"],
    ["14","Wav2Vec 2.0 Fine-Tuning for PD Detection (2024)","Pre-trained Wav2Vec 2.0","Multiple Datasets Combined","86.8%"],
    ["15","Jitter-Shimmer Analysis with Deep Neural Networks (2022)","Fully Connected DNN on Voice Features","PC-GITA","81.4%"],
    ["16","Capsule Networks for PD Voice Classification (2023)","CapsNet on MFCC + Prosody Features","mPower + PC-GITA","84.5%"],
  ];

  spRows.forEach((row, ri) => {
    const ry = 1.15 + ri * 0.545;
    const bg = ri % 2 === 0 ? "0D2626" : "0A2020";
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 9.78, h: 0.52, fill: { color: bg }, line: { color: bg } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 0.56, h: 0.52, fill: { color: ri % 2 === 0 ? "073B3B" : "052E2E" }, line: { color: "0E5555" } });
    s.addText(row[0], { x: 0.22, y: ry, w: 0.56, h: 0.52, fontSize: 11, color: "5EEAD4", bold: true, align: "center", valign: "middle" });
    s.addText(row[1], { x: 0.84, y: ry+0.04, w: 3.28, h: 0.44, fontSize: 9, color: C.white, valign: "middle" });
    s.addText(row[2], { x: 4.18, y: ry+0.04, w: 2.24, h: 0.44, fontSize: 9, color: "5EEAD4", bold: true, valign: "middle" });
    s.addText(row[3], { x: 6.48, y: ry+0.04, w: 2.18, h: 0.44, fontSize: 9, color: "CBD5E1", valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.72, y: ry+0.1, w: 1.06, h: 0.32, fill: { color: "052E16" }, line: { color: "166534" } });
    s.addText(row[4], { x: 8.72, y: ry+0.1, w: 1.06, h: 0.32, fontSize: 11, color: "4ADE80", bold: true, align: "center", valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 0.22, y: ry+0.52, w: 9.78, h: 0, line: { color: "0E5555", width: 0.5 } });
  });
  [0.78, 4.12, 6.48, 8.72].forEach(x => {
    s.addShape(pres.shapes.LINE, { x, y: 0.77, w: 0, h: 4.51, line: { color: "0E5555", width: 0.5 } });
  });

  s.addText("Slide 7  •  Multi-Modal Parkinson's Disease Detection System", { x: 0.28, y: 5.47, w: 9.44, h: 0.15, fontSize: 8, color: "2A6B6B", align: "left" });
}

// =============================================
// SLIDE 8 — LITERATURE: Multimodal (FULL PAGE BEAUTIFUL)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "0D1B3E" };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: C.blue }, line: { color: C.blue } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0, w: 9.78, h: 0.72, fill: { color: "0A1B42" }, line: { color: "0A1B42" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 0.72, w: 9.78, h: 0.05, fill: { color: C.blue }, line: { color: C.blue } });

  s.addText("TABLES 3 & 4: LITERATURE SURVEY", { x: 6.4, y: 0.12, w: 3.5, h: 0.22, fontSize: 9, color: "93C5FD", bold: true, align: "right", charSpacing: 1 });
  s.addText("MM", { x: 0.32, y: 0.06, w: 0.68, h: 0.58, fontSize: 13, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("Existing Systems: Multi-Modal & Recent Approaches", { x: 1.1, y: 0.1, w: 8.6, h: 0.52, fontSize: 21, color: C.white, bold: true, fontFace: "Calibri" });

  const hCols = [0.22, 0.78, 4.0, 6.36, 8.72];
  const hWidths = [0.56, 3.22, 2.36, 2.36, 1.06];
  const hLabels = ["Sl", "Title & Year", "Methodology", "Dataset", "Acc"];
  hCols.forEach((x, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.77, w: hWidths[i], h: 0.38, fill: { color: "0D2460" }, line: { color: "0D2460" } });
    s.addText(hLabels[i], { x: x+0.06, y: 0.77, w: hWidths[i]-0.08, h: 0.38, fontSize: 10.5, color: C.white, bold: true, align: i===4?"center":"left", valign: "middle" });
  });

  const mmRows = [
    ["17","Multi-Modal PD Detection: Gait + Speech (2023)","Late Fusion CNN + LSTM","Custom Multi-Modal Dataset","91.2%"],
    ["18","Handwriting + MRI Fusion for PD Diagnosis (2022)","Early Fusion ResNet + 3D-CNN","Clinical Multi-Modal Data","90.7%"],
    ["19","Speech + Facial Expression Analysis for PD (2024)","Attention-Based Multi-Modal Fusion","Video + Audio (320 patients)","93.1%"],
    ["20","Triple-Modal: Gait + Speech + Handwriting (2023)","Hierarchical Fusion Network","Custom (280 subjects)","94.3%"],
    ["21","Meta-Learning for Multi-Modal PD Classification (2024)","MAML on Speech + Handwriting","Combined Datasets","92.8%"],
    ["22","Graph Neural Network for Multi-Feature PD (2023)","GNN on Feature Correlation Graph","Multi-Modal Clinical Data","91.5%"],
    ["23","Federated Learning for Privacy-Preserving PD (2024)","Federated ResNet on Handwriting","Distributed (1,500 samples)","88.8%"],
    ["24","Self-Supervised Learning for PD Speech (2023)","SimCLR Pre-training + Fine-tuning","Unlabeled + Labeled Speech","86.4%"],
  ];

  mmRows.forEach((row, ri) => {
    const ry = 1.15 + ri * 0.488;
    const bg = ri % 2 === 0 ? "122050" : "0F1A42";
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 9.78, h: 0.47, fill: { color: bg }, line: { color: bg } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: ry, w: 0.56, h: 0.47, fill: { color: ri % 2 === 0 ? "0D2460" : "091A48" }, line: { color: "1E3A6A" } });
    s.addText(row[0], { x: 0.22, y: ry, w: 0.56, h: 0.47, fontSize: 11, color: "93C5FD", bold: true, align: "center", valign: "middle" });
    s.addText(row[1], { x: 0.84, y: ry+0.03, w: 3.16, h: 0.41, fontSize: 8.5, color: C.white, valign: "middle" });
    s.addText(row[2], { x: 4.06, y: ry+0.03, w: 2.24, h: 0.41, fontSize: 8.5, color: "93C5FD", bold: true, valign: "middle" });
    s.addText(row[3], { x: 6.42, y: ry+0.03, w: 2.24, h: 0.41, fontSize: 8.5, color: "CBD5E1", valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.72, y: ry+0.08, w: 1.06, h: 0.3, fill: { color: "052E16" }, line: { color: "166534" } });
    s.addText(row[4], { x: 8.72, y: ry+0.08, w: 1.06, h: 0.3, fontSize: 11, color: "4ADE80", bold: true, align: "center", valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: 0.22, y: ry+0.47, w: 9.78, h: 0, line: { color: "1E3A6A", width: 0.5 } });
  });
  [0.78, 4.0, 6.36, 8.72].forEach(x => {
    s.addShape(pres.shapes.LINE, { x, y: 0.77, w: 0, h: 4.12, line: { color: "1E3A6A", width: 0.5 } });
  });

  // Our result highlight bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.22, y: 5.06, w: 9.78, h: 0.36, fill: { color: "0B2252" }, line: { color: C.teal, width: 1.5 } });
  s.addText("STAR  Ours (CMAFN):  Handwriting + Speech  via  Bidirectional Cross-Attention + GMU  =>  96.94% Accuracy  |  AUC 0.9995  (Patient-level 5-fold CV)", { x: 0.3, y: 5.08, w: 9.6, h: 0.32, fontSize: 9.5, color: C.teal, bold: true, align: "center", valign: "middle" });

  s.addText("Slide 8  •  Multi-Modal Parkinson's Disease Detection System", { x: 0.28, y: 5.47, w: 9.44, h: 0.15, fontSize: 8, color: "4A6FA5", align: "left" });
}

// =============================================
// SLIDE 9 — SYSTEM ARCHITECTURE
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Architecture", "System Architecture Overview", 26, "SYS");

  // Flow diagram
  const blocks = [
    { x: 0.32, y: 1.05, w: 1.8, h: 1.2, bg: C.blueLt, border: C.navy, label: "INPUT 1\nHandwriting\nSpiral/Wave\n224×224" },
    { x: 0.32, y: 2.5, w: 1.8, h: 1.2, bg: C.tealLt, border: C.teal, label: "INPUT 2\nSpeech Audio\n16 kHz\n< 8 seconds" },
    { x: 2.55, y: 1.0, w: 2.2, h: 1.35, bg: C.blueLt, border: C.navy, label: "MODULE A\nHandwriting Analysis\nMLP + EfficientNet-B0\n16 Biomarkers\nAcc ≈ 93.6%" },
    { x: 2.55, y: 2.45, w: 2.2, h: 1.35, bg: C.tealLt, border: C.teal, label: "MODULE B\nSpeech Analysis\nXLS-R + BiLSTM\nMFCC + Praat\nAcc ≈ 91.7%" },
    { x: 5.2, y: 1.4, w: 2.4, h: 2.0, bg: C.blueLt, border: C.navy, label: "MODULE C\nCMAFN Fusion\nBidirectional\nCross-Attention\n+ GMU\nAcc = 96.94%" },
    { x: 8.05, y: 1.65, w: 1.65, h: 1.5, bg: "1F2937", border: C.navy, label: "OUTPUT\nPD Risk:\nLOW\nMODERATE\nHIGH" },
  ];

  blocks.forEach(({ x, y, w, h, bg, border, label }) => {
    addCard(s, x, y, w, h, bg, border);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.06, fill: { color: border }, line: { color: border } });
    const isOutput = bg === "1F2937";
    s.addText(label, { x: x + 0.08, y: y + 0.1, w: w - 0.16, h: h - 0.18, fontSize: 9.5, color: isOutput ? C.white : C.navy, align: "center", valign: "middle", fontFace: "Calibri" });
  });

  // Arrows
  [[2.12, 1.52], [2.12, 2.97], [4.75, 1.82], [4.75, 3.02], [7.6, 2.35]].forEach(([x, y]) => {
    s.addShape(pres.shapes.LINE, { x, y, w: 0.4, h: 0, line: { color: C.gray, width: 1.5 } });
  });

  // Labels
  s.addText("Non-invasive\nInputs", { x: 0.27, y: 3.85, w: 1.9, h: 0.4, fontSize: 9, color: C.teal, bold: true, align: "center" });
  s.addText("Unimodal\nModules", { x: 2.5, y: 3.85, w: 2.3, h: 0.4, fontSize: 9, color: C.navy, bold: true, align: "center" });
  s.addText("Cross-Modal\nFusion", { x: 5.15, y: 3.5, w: 2.5, h: 0.4, fontSize: 9, color: C.navy, bold: true, align: "center" });
  s.addText("Clinical\nDecision", { x: 8.0, y: 3.25, w: 1.7, h: 0.4, fontSize: 9, color: C.gray, bold: true, align: "center" });

  // Bottom info bar
  const infoItems = [["Flask Web App", "Real-time inference"], ["CPU-friendly", "< 3s end-to-end"], ["Patient-level CV", "5-fold, no leakage"], ["~85 MB model", "INT8 quantized"]];
  infoItems.forEach(([t, s2], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32 + i * 2.4, y: 4.5, w: 2.2, h: 0.68, fill: { color: C.grayLt }, line: { color: C.grayMd } });
    s.addText(t, { x: 0.42 + i * 2.4, y: 4.55, w: 2.0, h: 0.24, fontSize: 10.5, color: C.navy, bold: true });
    s.addText(s2, { x: 0.42 + i * 2.4, y: 4.78, w: 2.0, h: 0.32, fontSize: 9, color: C.gray });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 10 — MODULE A: HANDWRITING (REDESIGNED)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "F8FAFF" };

  // Top navy band
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.navy }, line: { color: C.navy } });
  // Left accent rail
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.08, w: 0.18, h: 3.6, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.68, w: 0.18, h: 1.2, fill: { color: C.blue }, line: { color: C.blue } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.88, w: 0.18, h: 0.75, fill: { color: C.teal }, line: { color: C.teal } });

  // Header area
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("A", { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fontSize: 16, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("MODULE A", { x: 0.78, y: 0.1, w: 2, h: 0.2, fontSize: 8, color: C.blue, bold: true, charSpacing: 3 });
  s.addText("Handwriting Analysis", { x: 0.78, y: 0.28, w: 6.5, h: 0.32, fontSize: 24, color: C.dark, bold: true, fontFace: "Calibri" });

  // Metric badges
  s.addShape(pres.shapes.RECTANGLE, { x: 7.5, y: 0.08, w: 1.22, h: 0.52, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("93.6%", { x: 7.5, y: 0.08, w: 1.22, h: 0.3, fontSize: 18, color: C.white, bold: true, align: "center" });
  s.addText("Accuracy", { x: 7.5, y: 0.36, w: 1.22, h: 0.2, fontSize: 8, color: "93C5FD", align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.76, y: 0.08, w: 1.22, h: 0.52, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("0.985", { x: 8.76, y: 0.08, w: 1.22, h: 0.3, fontSize: 18, color: C.white, bold: true, align: "center" });
  s.addText("AUC", { x: 8.76, y: 0.36, w: 1.22, h: 0.2, fontSize: 8, color: "99F6E4", align: "center" });

  // ── LEFT: FLOWCHART ──────────────────────────────────────────────────────
  // Card background
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 4.6, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 4.6, h: 0.32, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("ARCHITECTURE  —  Fig. A", { x: 0.38, y: 0.78, w: 4.34, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  // Node drawing helper: rounded rectangle + label
  const N = (cx, cy, w, h, line1, line2, fillC, strokeC, textC) => {
    const makeSh = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.10 });
    s.addShape(pres.shapes.RECTANGLE, { x: cx-w/2, y: cy-h/2, w, h, fill: { color: fillC }, line: { color: strokeC, width: 1.8 }, shadow: makeSh() });
    if (line2) {
      s.addText(line1, { x: cx-w/2+0.06, y: cy-h/2+0.03, w: w-0.12, h: h*0.48, fontSize: 9.5, color: textC, bold: true, align: "center", valign: "bottom" });
      s.addText(line2, { x: cx-w/2+0.06, y: cy, w: w-0.12, h: h*0.45, fontSize: 8.5, color: textC, bold: false, align: "center", valign: "top", italic: true });
    } else {
      s.addText(line1, { x: cx-w/2+0.06, y: cy-h/2, w: w-0.12, h, fontSize: 10, color: textC, bold: true, align: "center", valign: "middle" });
    }
  };
  // Arrow helper
  const AR = (x1, y1, x2, y2, col) => {
    const col2 = col || "334455";
    s.addShape(pres.shapes.LINE, { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1)||0.001, h: Math.abs(y2-y1)||0.001, line: { color: col2, width: 1.6, endArrowType: "triangle" } });
  };

  const fc = 2.55;
  // Node 1 — input
  N(fc, 1.18, 3.4, 0.42, "Handwriting Image", "(Spiral / Wave Drawing)", C.navy, C.navy, C.white);
  AR(fc, 1.39, fc, 1.62, C.navy);
  // Node 2 — biomarkers
  N(fc, 1.82, 2.9, 0.38, "16 Spatial Biomarkers", "OpenCV feature extraction", C.blueLt, C.navy, C.navy);
  // split: down, then horizontal branches, then arrows down to nodes
  s.addShape(pres.shapes.LINE, { x: fc, y: 2.01, w: 0.001, h: 0.1, line: { color: "334455", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc-0.95, y: 2.11, w: 0.95, h: 0, line: { color: "334455", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc, y: 2.11, w: 0.95, h: 0, line: { color: "006655", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc-0.95, y: 2.11, w: 0.001, h: 0.15, line: { color: "334455", width: 1.6, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: fc+0.95, y: 2.11, w: 0.001, h: 0.15, line: { color: "006655", width: 1.6, endArrowType: "triangle" } });
  // Node 3L — MLP
  s.addShape(pres.shapes.RECTANGLE, { x: fc-0.95-0.85, y: 2.26, w: 1.7, h: 0.56, fill: { color: "EEF4FF" }, line: { color: C.navy, width: 1.8 }, shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 } });
  s.addText("MLP Path", { x: fc-0.95-0.85+0.06, y: 2.26, w: 1.58, h: 0.3, fontSize: 10.5, color: C.navy, bold: true, align: "center", valign: "middle" });
  s.addText("16 engineered features", { x: fc-0.95-0.85+0.06, y: 2.52, w: 1.58, h: 0.26, fontSize: 8, color: C.gray, align: "center", italic: true });
  // Node 3R — CNN
  s.addShape(pres.shapes.RECTANGLE, { x: fc+0.95-0.9, y: 2.26, w: 1.8, h: 0.56, fill: { color: "E6FAF8" }, line: { color: C.teal, width: 1.8 }, shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 } });
  s.addText("CNN Path", { x: fc+0.95-0.9+0.06, y: 2.26, w: 1.68, h: 0.3, fontSize: 10.5, color: C.teal, bold: true, align: "center", valign: "middle" });
  s.addText("EfficientNet-B0 + CBAM", { x: fc+0.95-0.9+0.06, y: 2.52, w: 1.68, h: 0.26, fontSize: 8, color: C.gray, align: "center", italic: true });
  // Merge: vertical down from each branch, horizontal to center, vertical to LR node
  s.addShape(pres.shapes.LINE, { x: fc-0.95, y: 2.82, w: 0.001, h: 0.14, line: { color: "334455", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc+0.95, y: 2.82, w: 0.001, h: 0.14, line: { color: "006655", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc-0.95, y: 2.96, w: 0.95, h: 0, line: { color: "334455", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc, y: 2.96, w: 0.95, h: 0, line: { color: "006655", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: fc, y: 2.96, w: 0.001, h: 0.12, line: { color: "334455", width: 1.6, endArrowType: "triangle" } });
  // Node 4 — Logistic Regression
  s.addShape(pres.shapes.RECTANGLE, { x: fc-1.4, y: 3.08, w: 2.8, h: 0.5, fill: { color: "F0F4FF" }, line: { color: C.navy, width: 1.8 }, shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 } });
  s.addText("Logistic Regression", { x: fc-1.4+0.08, y: 3.08, w: 2.64, h: 0.3, fontSize: 11, color: C.navy, bold: true, align: "center", valign: "middle" });
  s.addText("Meta-learner stacking", { x: fc-1.4+0.08, y: 3.33, w: 2.64, h: 0.22, fontSize: 8, color: C.gray, align: "center", italic: true });
  AR(fc, 3.58, fc, 3.82, C.navy);
  // Node 5 — result
  s.addShape(pres.shapes.RECTANGLE, { x: fc-1.45, y: 3.82, w: 2.9, h: 0.55, fill: { color: C.navy }, line: { color: C.navy, width: 2 } });
  s.addText("93.6%  Accuracy", { x: fc-1.45, y: 3.82, w: 1.6, h: 0.55, fontSize: 17, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("AUC  0.985", { x: fc+0.16, y: 3.82, w: 1.28, h: 0.55, fontSize: 11, color: "4ADE80", bold: true, align: "center", valign: "middle" });
  // caption
  s.addText("Patient-level stratified 5-fold cross-validation  •  No data leakage", { x: 0.35, y: 4.44, w: 4.4, h: 0.2, fontSize: 8.5, color: C.gray, align: "center", italic: true });

  // ── RIGHT: BIOMARKERS ────────────────────────────────────────────────────
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 0.76, w: 4.75, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 0.76, w: 4.75, h: 0.32, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("16 SPATIAL BIOMARKERS", { x: 5.13, y: 0.78, w: 4.5, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  const bGroups = [
    ["01", "Stroke & Thickness", "Stroke width mean  •  Stroke width std  •  Width variability", C.navy, "EEF4FF"],
    ["02", "Geometry & Smoothness", "Curvature mean  •  Direction changes  •  Contour roughness", C.teal, "E6FAF8"],
    ["03", "Density & Components", "Connected component density  •  Ink density  •  Edge density", C.navy, "EEF4FF"],
    ["04", "Statistics & Complexity", "Entropy  •  Fractal dimension  •  Hu moments (1–7)", C.teal, "E6FAF8"],
  ];
  bGroups.forEach(([num, title, items, tc, bg], i) => {
    const y = 1.16 + i * 1.02;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.55, h: 0.9, fill: { color: bg }, line: { color: tc, width: 1 } });
    // number badge
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 0.42, h: 0.9, fill: { color: tc }, line: { color: tc } });
    s.addText(num, { x: 5.1, y, w: 0.42, h: 0.9, fontSize: 13, color: C.white, bold: true, align: "center", valign: "middle" });
    s.addText(title, { x: 5.6, y: y + 0.1, w: 3.95, h: 0.28, fontSize: 11.5, color: tc, bold: true });
    s.addText(items, { x: 5.6, y: y + 0.42, w: 3.95, h: 0.4, fontSize: 9.5, color: C.textMd });
  });

  // takeaway bar
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 5.24, w: 4.75, h: 0.18, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("Captures tremor, micrographia & motor control degradation — portable, non-invasive input", { x: 5.05, y: 5.24, w: 4.65, h: 0.18, fontSize: 7.5, color: "93C5FD", bold: true, align: "center", valign: "middle" });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.25, y: 5.44, w: 9.5, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 11 — MODULE B: SPEECH (REDESIGNED)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "F6FEFF" };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.08, w: 0.18, h: 3.6, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.68, w: 0.18, h: 1.2, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.88, w: 0.18, h: 0.75, fill: { color: "0D9488" }, line: { color: "0D9488" } });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("B", { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fontSize: 16, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("MODULE B", { x: 0.78, y: 0.1, w: 2, h: 0.2, fontSize: 8, color: C.teal, bold: true, charSpacing: 3 });
  s.addText("Speech Analysis", { x: 0.78, y: 0.28, w: 6.5, h: 0.32, fontSize: 24, color: C.dark, bold: true, fontFace: "Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.5, y: 0.08, w: 1.22, h: 0.52, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("91.7%", { x: 7.5, y: 0.08, w: 1.22, h: 0.3, fontSize: 18, color: C.white, bold: true, align: "center" });
  s.addText("Accuracy", { x: 7.5, y: 0.36, w: 1.22, h: 0.2, fontSize: 8, color: "99F6E4", align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.76, y: 0.08, w: 1.22, h: 0.52, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("0.971", { x: 8.76, y: 0.08, w: 1.22, h: 0.3, fontSize: 18, color: C.white, bold: true, align: "center" });
  s.addText("AUC", { x: 8.76, y: 0.36, w: 1.22, h: 0.2, fontSize: 8, color: "93C5FD", align: "center" });

  // ── LEFT: FLOWCHART ──────────────────────────────────────────────────────
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 4.6, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 4.6, h: 0.32, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("ARCHITECTURE  —  Fig. B", { x: 0.38, y: 0.78, w: 4.34, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  const makeSh2 = () => ({ type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.09 });
  const AR2 = (x1, y1, x2, y2, col) => {
    s.addShape(pres.shapes.LINE, { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1)||0.001, h: Math.abs(y2-y1)||0.001, line: { color: col||"334455", width: 1.5, endArrowType: "triangle" } });
  };

  const sc = 2.55;
  // Input node
  s.addShape(pres.shapes.RECTANGLE, { x: sc-1.55, y: 1.14, w: 3.1, h: 0.44, fill: { color: C.teal }, line: { color: C.teal, width: 2 }, shadow: makeSh2() });
  s.addText("Speech Audio  (16 kHz, < 8s)", { x: sc-1.55+0.08, y: 1.14, w: 2.94, h: 0.44, fontSize: 10.5, color: C.white, bold: true, align: "center", valign: "middle" });

  const pxs = [sc-1.5, sc-0.5, sc+0.5, sc+1.5];
  // fan-out: straight down, horizontal bus, then vertical drops to each node
  s.addShape(pres.shapes.LINE, { x: sc, y: 1.58, w: 0.001, h: 0.1, line: { color: "0D9488", width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: sc-1.5, y: 1.68, w: 3.0, h: 0, line: { color: "0D9488", width: 1.6 } });
  pxs.forEach(px => s.addShape(pres.shapes.LINE, { x: px, y: 1.68, w: 0.001, h: 0.16, line: { color: "0D9488", width: 1.6, endArrowType: "triangle" } }));

  // 4 pathway nodes
  const pDefs = [
    ["XLS-R", "BiLSTM", "E6FAF8", C.teal, "006655"],
    ["Mel-Spec", "CNN", "EEF4FF", C.navy, "0B2A6F"],
    ["MFCC", "+ Delta", "E6FAF8", C.teal, "006655"],
    ["Praat", "Acoustic", "EEF4FF", C.navy, "0B2A6F"],
  ];
  pxs.forEach((px, i) => {
    const [l1, l2, bg, tc, border] = pDefs[i];
    s.addShape(pres.shapes.RECTANGLE, { x: px-0.55, y: 1.84, w: 1.1, h: 0.56, fill: { color: bg }, line: { color: border, width: 1.8 }, shadow: makeSh2() });
    s.addText(l1, { x: px-0.55+0.04, y: 1.84, w: 1.02, h: 0.3, fontSize: 10, color: tc, bold: true, align: "center", valign: "middle" });
    s.addText(l2, { x: px-0.55+0.04, y: 2.1, w: 1.02, h: 0.26, fontSize: 8.5, color: C.gray, align: "center", italic: true });
  });

  // fan-in: vertical up from each node, horizontal to center, arrow down into cross-attention
  pxs.forEach(px => s.addShape(pres.shapes.LINE, { x: px, y: 2.4, w: 0.001, h: 0.14, line: { color: "334455", width: 1.5 } }));
  s.addShape(pres.shapes.LINE, { x: sc-1.5, y: 2.54, w: 3.0, h: 0, line: { color: "334455", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc, y: 2.54, w: 0.001, h: 0.06, line: { color: "334455", width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sc-1.5, y: 2.6, w: 3.0, h: 0.46, fill: { color: "EEF4FF" }, line: { color: C.navy, width: 2 }, shadow: makeSh2() });
  s.addText("Multi-Head Cross-Attention", { x: sc-1.5+0.08, y: 2.6, w: 2.84, h: 0.46, fontSize: 11, color: C.navy, bold: true, align: "center", valign: "middle" });

  // split to DL / Classical ML via T-junction
  s.addShape(pres.shapes.LINE, { x: sc, y: 3.06, w: 0.001, h: 0.11, line: { color: "334455", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc-0.72, y: 3.17, w: 1.44, h: 0, line: { color: "334455", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc-0.72, y: 3.17, w: 0.001, h: 0.11, line: { color: "334455", width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: sc+0.72, y: 3.17, w: 0.001, h: 0.11, line: { color: "006655", width: 1.5, endArrowType: "triangle" } });

  s.addShape(pres.shapes.RECTANGLE, { x: sc-1.5, y: 3.28, w: 1.44, h: 0.46, fill: { color: "EEF4FF" }, line: { color: C.navy, width: 1.8 }, shadow: makeSh2() });
  s.addText("Deep Learning", { x: sc-1.5+0.06, y: 3.28, w: 1.32, h: 0.46, fontSize: 10, color: C.navy, bold: true, align: "center", valign: "middle" });
  s.addShape(pres.shapes.RECTANGLE, { x: sc+0.06, y: 3.28, w: 1.44, h: 0.46, fill: { color: "E6FAF8" }, line: { color: C.teal, width: 1.8 }, shadow: makeSh2() });
  s.addText("Classical ML", { x: sc+0.06+0.06, y: 3.28, w: 1.32, h: 0.46, fontSize: 10, color: C.teal, bold: true, align: "center", valign: "middle" });

  // merge DL+ML → ridge LR via join
  s.addShape(pres.shapes.LINE, { x: sc-0.72, y: 3.74, w: 0.001, h: 0.13, line: { color: "334455", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc+0.72, y: 3.74, w: 0.001, h: 0.13, line: { color: "006655", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc-0.72, y: 3.87, w: 1.44, h: 0, line: { color: "334455", width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: sc, y: 3.87, w: 0.001, h: 0.09, line: { color: "334455", width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.RECTANGLE, { x: sc-1.45, y: 3.96, w: 2.9, h: 0.44, fill: { color: "F0F4FF" }, line: { color: C.navy, width: 1.8 }, shadow: makeSh2() });
  s.addText("Ridge Logistic Regression", { x: sc-1.45+0.08, y: 3.96, w: 2.74, h: 0.44, fontSize: 10.5, color: C.navy, bold: true, align: "center", valign: "middle" });
  AR2(sc, 4.4, sc, 4.62, C.teal);
  // result
  s.addShape(pres.shapes.RECTANGLE, { x: sc-1.45, y: 4.62, w: 2.9, h: 0.52, fill: { color: C.teal }, line: { color: C.teal, width: 2 } });
  s.addText("91.7%  Accuracy", { x: sc-1.45, y: 4.62, w: 1.7, h: 0.52, fontSize: 17, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("AUC  0.971", { x: sc+0.25, y: 4.62, w: 1.18, h: 0.52, fontSize: 11, color: "CCFBF1", bold: true, align: "center", valign: "middle" });
  s.addText("Fig. B — 4-pathway speech module with cross-attention fusion", { x: 0.35, y: 5.22, w: 4.4, h: 0.18, fontSize: 8, color: C.gray, align: "center", italic: true });

  // ── RIGHT: PATHWAY DETAILS ───────────────────────────────────────────────
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 0.76, w: 4.75, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 0.77, w: 4.75, h: 0.32, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("PATHWAY DETAILS", { x: 5.13, y: 0.78, w: 4.5, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  const spPths = [
    ["01", "XLS-R + BiLSTM", "Self-supervised wav2vec 2.0 multilingual (300M). Attentive pooling + BiLSTM captures phonetic, prosodic and articulation patterns far beyond manual features.", C.teal, "E6FAF8"],
    ["02", "Mel-Spectrogram CNN", "ResNet-based 2D CNN processes log-mel spectrogram image. Captures spectral energy distribution over time — key indicator of reduced vocal control.", C.navy, "EEF4FF"],
    ["03", "MFCC + Derivatives", "13–40 Mel-Frequency Cepstral Coefficients + 1st/2nd delta features. Mean, variance, min, max statistical descriptors per frame summarize timbre & resonance.", C.teal, "E6FAF8"],
    ["04", "Praat Acoustic", "Voice quality measures: Jitter (freq variability), Shimmer (amplitude variability), HNR, NHR, DUV. Clinically established dysarthria and hypophonia markers.", C.navy, "EEF4FF"],
  ];
  spPths.forEach(([num, title, body, tc, bg], i) => {
    const y = 1.16 + i * 1.02;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.55, h: 0.9, fill: { color: bg }, line: { color: tc, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 0.42, h: 0.9, fill: { color: tc }, line: { color: tc } });
    s.addText(num, { x: 5.1, y, w: 0.42, h: 0.9, fontSize: 13, color: C.white, bold: true, align: "center", valign: "middle" });
    s.addText(title, { x: 5.6, y: y + 0.08, w: 3.95, h: 0.26, fontSize: 11.5, color: tc, bold: true });
    s.addText(body, { x: 5.6, y: y + 0.38, w: 3.95, h: 0.48, fontSize: 9, color: C.textMd });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 5.24, w: 4.75, h: 0.18, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("XLS-R + cross-attention fusion outperforms single-pathway speech models by a wide margin", { x: 5.05, y: 5.24, w: 4.65, h: 0.18, fontSize: 7.5, color: C.white, bold: true, align: "center", valign: "middle" });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.25, y: 5.44, w: 9.5, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 12 — MODULE C: CMAFN (REDESIGNED)
// =============================================
{
  const s = pres.addSlide();
  s.background = { color: "F8FAFF" };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.08, w: 0.18, h: 3.6, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.68, w: 0.18, h: 1.2, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.88, w: 0.18, h: 0.75, fill: { color: C.blue }, line: { color: C.blue } });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("C", { x: 0.25, y: 0.1, w: 0.42, h: 0.42, fontSize: 16, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("MODULE C", { x: 0.78, y: 0.1, w: 2.5, h: 0.2, fontSize: 8, color: C.blue, bold: true, charSpacing: 3 });
  s.addText("Cross-Modal Attention Fusion Network (CMAFN)", { x: 0.78, y: 0.28, w: 8.5, h: 0.32, fontSize: 20, color: C.dark, bold: true, fontFace: "Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.88, y: 0.08, w: 1.1, h: 0.52, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("96.94%", { x: 7.88, y: 0.08, w: 1.1, h: 0.3, fontSize: 14, color: C.white, bold: true, align: "center" });
  s.addText("Accuracy", { x: 7.88, y: 0.36, w: 1.1, h: 0.2, fontSize: 7.5, color: "93C5FD", align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.0, y: 0.08, w: 0.96, h: 0.52, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("0.9995", { x: 9.0, y: 0.08, w: 0.96, h: 0.3, fontSize: 13, color: C.white, bold: true, align: "center" });
  s.addText("AUC", { x: 9.0, y: 0.36, w: 0.96, h: 0.2, fontSize: 7.5, color: "99F6E4", align: "center" });

  // ── LEFT: CMAFN FLOWCHART ────────────────────────────────────────────────
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 5.25, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.76, w: 5.25, h: 0.32, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("CMAFN ARCHITECTURE  —  Fig. C", { x: 0.38, y: 0.78, w: 5.02, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  const makeSh3 = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.10 });
  const CaN = (cx, cy, w, h, l1, l2, fillC, strokeC, textC) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx-w/2, y: cy-h/2, w, h, fill: { color: fillC }, line: { color: strokeC, width: 1.8 }, shadow: makeSh3() });
    s.addText(l1, { x: cx-w/2+0.06, y: cy-h/2+0.02, w: w-0.12, h: h*0.5, fontSize: 9.5, color: textC, bold: true, align: "center", valign: "middle" });
    if (l2) s.addText(l2, { x: cx-w/2+0.06, y: cy, w: w-0.12, h: h*0.44, fontSize: 8, color: textC, align: "center", valign: "top", italic: true });
  };
  const CaAR = (x1, y1, x2, y2, col) => {
    s.addShape(pres.shapes.LINE, { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1)||0.001, h: Math.abs(y2-y1)||0.001, line: { color: col||"334455", width: 1.6, endArrowType: "triangle" } });
  };

  const hw = 1.5, sp = 4.25;
  // Row 1 - Encoders
  CaN(hw, 1.26, 2.1, 0.44, "Handwriting Encoder", "(EfficientNet-B4)", C.white, C.navy, C.navy);
  CaN(sp, 1.26, 2.1, 0.44, "Speech Encoder", "(XLS-R  300M)", C.white, C.teal, C.teal);
  CaAR(hw, 1.48, hw, 1.74, C.navy);
  CaAR(sp, 1.48, sp, 1.74, C.teal);
  // Row 2 - Projections
  CaN(hw, 1.94, 1.85, 0.4, "Projection", "512 → 256-d", "EEF4FF", C.navy, C.navy);
  CaN(sp, 1.94, 1.85, 0.4, "Projection", "960 → 256-d", "E6FAF8", C.teal, C.teal);
  // dashed cross-connection between projections
  s.addShape(pres.shapes.LINE, { x: hw+0.93, y: 1.94, w: sp-hw-1.86, h: 0, line: { color: "AAAAAA", width: 1, dashType: "sysDash" } });
  CaAR(hw, 2.14, hw, 2.42, C.navy);
  CaAR(sp, 2.14, sp, 2.42, C.teal);
  // Row 3 - Cross-Attention
  CaN(hw, 2.62, 1.95, 0.38, "HW Cross-Attention", null, "EEF4FF", C.navy, C.navy);
  CaN(sp, 2.62, 2.1, 0.38, "Speech Cross-Attention", null, "E6FAF8", C.teal, C.teal);
  // Bidirectional arrows (the signature of CMAFN)
  s.addShape(pres.shapes.RECTANGLE, { x: hw+0.98, y: 2.52, w: sp-hw-1.96, h: 0.22, fill: { color: "F5F5F5" }, line: { color: "CCCCCC", width: 0.5 } });
  s.addShape(pres.shapes.LINE, { x: hw+0.98, y: 2.58, w: sp-hw-1.96, h: 0, line: { color: C.navy, width: 1.6, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: hw+0.98, y: 2.65, w: sp-hw-1.96, h: 0, line: { color: C.teal, width: 1.6, endArrowType: "open", beginArrowType: "triangle" } });
  s.addText("Bidirectional", { x: (hw+sp)/2-0.38, y: 2.74, w: 0.76, h: 0.16, fontSize: 7, color: "666666", bold: true, align: "center", italic: true });

  // Both down to GMU (orthogonal routing)
  const gmucx = (hw+sp)/2;
  s.addShape(pres.shapes.LINE, { x: hw, y: 2.81, w: 0.001, h: 0.18, line: { color: C.navy, width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: sp, y: 2.81, w: 0.001, h: 0.18, line: { color: C.teal, width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: hw, y: 2.99, w: gmucx - hw, h: 0, line: { color: C.navy, width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: gmucx, y: 2.99, w: sp - gmucx, h: 0, line: { color: C.teal, width: 1.6 } });
  s.addShape(pres.shapes.LINE, { x: gmucx, y: 2.99, w: 0.001, h: 0.09, line: { color: C.navy, width: 1.6, endArrowType: "triangle" } });
  // Row 4 - GMU (orange/amber to differentiate)
  s.addShape(pres.shapes.RECTANGLE, { x: gmucx-1.18, y: 3.08, w: 2.36, h: 0.44, fill: { color: "FFF7ED" }, line: { color: "D97706", width: 2 }, shadow: makeSh3() });
  s.addText("Gated Multimodal Unit  (GMU)", { x: gmucx-1.18+0.08, y: 3.08, w: 2.2, h: 0.28, fontSize: 11, color: "92400E", bold: true, align: "center", valign: "middle" });
  s.addText("Learnable sigmoid gate balances HW + Speech", { x: gmucx-1.18+0.08, y: 3.33, w: 2.2, h: 0.17, fontSize: 7.5, color: "B45309", align: "center", italic: true });
  CaAR(gmucx, 3.52, gmucx, 3.78, "92400E");
  // Row 5 - Concat + FC
  s.addShape(pres.shapes.RECTANGLE, { x: gmucx-1.18, y: 3.78, w: 2.36, h: 0.4, fill: { color: "F5F5F5" }, line: { color: "555555", width: 1.6 }, shadow: makeSh3() });
  s.addText("Concatenate  +  FC Layer", { x: gmucx-1.18+0.08, y: 3.78, w: 2.2, h: 0.4, fontSize: 10.5, color: C.dark, bold: true, align: "center", valign: "middle" });
  CaAR(gmucx, 4.18, gmucx, 4.42, C.navy);
  // Row 6 - Result
  s.addShape(pres.shapes.RECTANGLE, { x: gmucx-1.25, y: 4.42, w: 2.5, h: 0.52, fill: { color: C.navy }, line: { color: C.navy, width: 2 } });
  s.addText("96.94% Accuracy", { x: gmucx-1.25, y: 4.42, w: 1.52, h: 0.52, fontSize: 14, color: C.white, bold: true, align: "center", valign: "middle" });
  s.addText("AUC 0.9995", { x: gmucx+0.27, y: 4.42, w: 0.96, h: 0.52, fontSize: 10, color: "4ADE80", bold: true, align: "center", valign: "middle" });
  s.addText("Bidirectional HW \u2194 Speech cross-attention + gating = best overall PD screening", { x: 0.35, y: 5.02, w: 5.05, h: 0.18, fontSize: 8, color: C.gray, align: "center", italic: true });

  // ── RIGHT: KEY DESIGN DECISIONS ──────────────────────────────────────────
  s.addShape(pres.shapes.RECTANGLE, { x: 5.65, y: 0.76, w: 4.1, h: 4.66, fill: { color: C.white }, line: { color: C.grayMd, width: 1 }, shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.08 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.65, y: 0.76, w: 4.1, h: 0.32, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("KEY DESIGN DECISIONS", { x: 5.78, y: 0.78, w: 3.88, h: 0.28, fontSize: 9.5, color: C.white, bold: true });

  const kpts = [
    ["01", "Bidirectional\nCross-Attention", "HW attends to Speech and Speech attends to HW. Each modality guides the other's feature selection via Transformer multi-head attention.", C.navy, "EEF4FF"],
    ["02", "Gated Multimodal\nUnit (GMU)", "Learnable sigmoid gate balances HW and Speech embedding contributions dynamically — prevents one modality from dominating.", "D97706", "FFF7ED"],
    ["03", "Modality\nDropout", "Each modality randomly masked (p=0.2) during training — model stays robust even when one input is missing at inference.", C.navy, "EEF4FF"],
    ["04", "MC Dropout\n(Uncertainty)", "Monte Carlo Dropout at inference gives calibrated confidence scores for clinical decision support & uncertain cases.", C.teal, "E6FAF8"],
  ];
  kpts.forEach(([num, title, body, tc, bg], i) => {
    const y = 1.16 + i * 1.02;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.75, y, w: 3.9, h: 0.9, fill: { color: bg }, line: { color: tc, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.75, y, w: 0.4, h: 0.9, fill: { color: tc }, line: { color: tc } });
    s.addText(num, { x: 5.75, y, w: 0.4, h: 0.9, fontSize: 13, color: C.white, bold: true, align: "center", valign: "middle" });
    const titleLines = title.split("\n");
    s.addText(titleLines[0], { x: 6.22, y: y + 0.06, w: 3.35, h: 0.22, fontSize: 11, color: tc, bold: true });
    if (titleLines[1]) s.addText(titleLines[1], { x: 6.22, y: y + 0.27, w: 3.35, h: 0.18, fontSize: 11, color: tc, bold: true });
    s.addText(body, { x: 6.22, y: y + (titleLines[1] ? 0.46 : 0.32), w: 3.35, h: 0.42, fontSize: 8.5, color: C.textMd });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.65, y: 5.24, w: 4.1, h: 0.18, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("+3.37% gain over best unimodal  —  complementary biomarkers + attention fusion wins", { x: 5.7, y: 5.24, w: 4.0, h: 0.18, fontSize: 7.5, color: "93C5FD", bold: true, align: "center", valign: "middle" });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.25, y: 5.44, w: 9.5, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 13 — DATASETS & TRAINING
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Experimental Setup", "Datasets & Training Strategy", 26, "DS");

  // Left: Datasets
  addCard(s, 0.32, 1.0, 4.55, 4.28);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 1.0, w: 4.55, h: 0.3, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("DATASETS", { x: 0.42, y: 1.03, w: 4.35, h: 0.24, fontSize: 10, color: C.white, bold: true, charSpacing: 2 });

  const dsets = [
    ["Handwriting Dataset", "3,264 images • 1,632 HC + 1,632 PD\nSpiral & Wave drawings (balanced classes)\nInput: 224×224 (HW module), 336×336 (fusion)\n16 spatial biomarkers extracted via OpenCV"],
    ["Speech Dataset", "Italian Parkinson's Voice & Speech (IEEE DataPort)\n831 recordings • 61 unique patients\nPatient-level group splits to avoid speaker leakage\nStandardized to 16 kHz, padded/trimmed to <8s"],
  ];
  dsets.forEach(([h, body], i) => {
    const y = 1.42 + i * 1.65;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.42, y, w: 4.35, h: 1.45, fill: { color: i === 0 ? C.blueLt : C.tealLt }, line: { color: C.grayMd } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.42, y, w: 4.35, h: 0.28, fill: { color: i === 0 ? C.navy : C.teal }, line: { color: i === 0 ? C.navy : C.teal } });
    s.addText(h, { x: 0.5, y: y + 0.02, w: 4.15, h: 0.24, fontSize: 10.5, color: C.white, bold: true });
    s.addText(body, { x: 0.5, y: y + 0.35, w: 4.15, h: 1.05, fontSize: 9.5, color: C.textMd });
  });

  // Right: Training
  addCard(s, 5.05, 1.0, 4.65, 4.28);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.05, y: 1.0, w: 4.65, h: 0.3, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("TRAINING STRATEGY", { x: 5.15, y: 1.03, w: 4.45, h: 0.24, fontSize: 10, color: C.white, bold: true, charSpacing: 2 });

  const tpts = [
    ["Optimizer & Schedule", "AdamW (LR: 3e-5 to 5e-5)\nOneCycleLR cosine annealing\nGradient clipping (max norm 1.0)\nEarly stopping (patience 10–12)"],
    ["Loss & Regularization", "Focal Loss (α=0.5–0.75, γ=2.0–3.0)\nLabel smoothing for calibration\nModalitydropout (fusion robustness)"],
    ["Data Augmentation", "Images: flip, rotation ±10°, brightness jitter,\nMixup, TTA (7 transforms)\nAudio: SpecAugment, VTLP, TTA (5×)"],
    ["Classical ML Stack", "RF, SVM, GBM, XGBoost, LightGBM\nSMOTE oversampling • PCA (99.5–99.9%)\nMeta-learner: Ridge LR stacking"],
  ];
  tpts.forEach(([h, body], i) => {
    const y = 1.42 + i * 0.95;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y, w: 4.45, h: 0.82, fill: { color: i % 2 === 0 ? C.grayLt : C.white }, line: { color: C.grayMd } });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y, w: 0.07, h: 0.82, fill: { color: i % 2 === 0 ? C.navy : C.teal }, line: { color: i % 2 === 0 ? C.navy : C.teal } });
    s.addText(h, { x: 5.3, y: y + 0.06, w: 4.2, h: 0.22, fontSize: 10.5, color: C.navy, bold: true });
    s.addText(body, { x: 5.3, y: y + 0.3, w: 4.2, h: 0.48, fontSize: 9, color: C.textMd });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 14 — RESULTS
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Evaluation", "Results & Performance Comparison", 26, "RES");

  // Headline banner
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 1.0, w: 9.38, h: 0.38, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("Fusion (CMAFN) achieves 96.94% accuracy & 0.9995 AUC-ROC — Patient-level stratified 5-fold CV (no data leakage)", { x: 0.42, y: 1.04, w: 9.18, h: 0.3, fontSize: 11, color: C.white, bold: true, align: "center" });

  // Module result cards
  const mods = [
    { label: "Module A — Handwriting", acc: "93.57%", auc: "0.985", bg: C.blueLt, tc: C.navy, detail: "MLP (16 biomarkers) + EfficientNet-B0 (CBAM)\nLogistic Regression meta-learner stacking" },
    { label: "Module B — Speech", acc: "91.70%", auc: "0.971", bg: C.tealLt, tc: C.teal, detail: "XLS-R BiLSTM + Mel CNN + MFCC + Praat\nMulti-head cross-attention + Ridge LR stacking" },
    { label: "Module C — CMAFN Fusion", acc: "96.94%", auc: "0.9995", bg: C.navy, tc: C.white, detail: "Bidirectional cross-attention + GMU gating\n+3.37% over best unimodal (handwriting)" },
  ];
  mods.forEach(({ label, acc, auc, bg, tc, detail }, i) => {
    const x = 0.32 + i * 3.18;
    addCard(s, x, 1.5, 3.05, 1.75, bg, bg);
    s.addText(label, { x: x + 0.12, y: 1.58, w: 2.82, h: 0.26, fontSize: 11, color: tc, bold: true });
    s.addText(acc, { x: x + 0.12, y: 1.84, w: 2.82, h: 0.56, fontSize: 30, color: tc, bold: true });
    s.addText("AUC  " + auc, { x: x + 0.12, y: 2.44, w: 2.82, h: 0.26, fontSize: 12, color: i === 2 ? "93C5FD" : C.gray, bold: true });
    s.addText(detail, { x: x + 0.12, y: 2.74, w: 2.82, h: 0.44, fontSize: 9, color: i === 2 ? "D1D5DB" : C.textMd });
  });

  // 5-fold stability table
  addCard(s, 0.32, 3.38, 4.55, 1.8);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 3.38, w: 4.55, h: 0.28, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("CMAFN — 5-Fold Balanced Accuracy Stability", { x: 0.42, y: 3.4, w: 4.35, h: 0.24, fontSize: 10, color: C.white, bold: true });

  const folds = [["Fold 1", "99.10%"], ["Fold 2", "96.40%"], ["Fold 3", "99.46%"], ["Fold 4", "99.82%"], ["Fold 5", "99.82%"]];
  folds.forEach(([fold, val], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.42 + i * 0.86, y: 3.75, w: 0.75, h: 0.38, fill: { color: C.navy }, line: { color: C.navy } });
    s.addText(fold, { x: 0.42 + i * 0.86, y: 3.77, w: 0.75, h: 0.16, fontSize: 7.5, color: "93C5FD", align: "center", bold: true });
    s.addText(val, { x: 0.42 + i * 0.86, y: 3.93, w: 0.75, h: 0.18, fontSize: 8.5, color: C.white, align: "center", bold: true });
  });
  s.addText("Mean ± Std: 98.92% ± 1.29%  (Balanced Accuracy across folds)", { x: 0.42, y: 4.22, w: 4.35, h: 0.28, fontSize: 9.5, color: C.teal, bold: true, align: "center" });
  s.addText("High fold consistency validates generalization under patient-level grouping.", { x: 0.42, y: 4.52, w: 4.35, h: 0.2, fontSize: 9, color: C.gray, align: "center" });

  // Bar chart
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy (%)", labels: ["Handwriting\n(Module A)", "Speech\n(Module B)", "CMAFN Fusion\n(Module C)"], values: [93.57, 91.70, 96.94] }
  ], {
    x: 5.05, y: 3.32, w: 4.6, h: 1.92,
    barDir: "col",
    chartColors: [C.navy, C.teal, "1E3A5F"],
    chartArea: { fill: { color: C.grayLt }, roundedCorners: false },
    catAxisLabelColor: C.textMd, valAxisLabelColor: C.textMd,
    valGridLine: { color: C.grayMd, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelColor: C.white, dataLabelFontBold: true,
    showLegend: false,
    valAxisMinVal: 88, valAxisMaxVal: 100
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 15 — KEY CONTRIBUTIONS
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Summary", "Key Contributions", 30, "KC");

  const contribs = [
    ["01", "16-Feature Handwriting Biomarker Framework", "Extracts clinically relevant spatial cues from spiral/wave drawings: stroke variability, roughness, entropy, fractal dimension, Hu moments, curvature, component density — providing interpretable motor impairment signals.", C.navy, C.blueLt],
    ["02", "Hybrid Speech Pipeline (DL + Classical ML)", "Combines self-supervised XLS-R representations with Mel-spectrogram CNN, MFCC, and Praat acoustic measures. Fused via multi-head cross-attention and stacked with Ridge Logistic Regression.", C.teal, C.tealLt],
    ["03", "Cross-Modal Attention Fusion Network (CMAFN)", "Novel bidirectional Transformer cross-attention (HW↔Speech) + Gated Multimodal Unit (GMU) + modality dropout. Learns complementary motor + vocal biomarker interactions for improved PD screening.", C.navy, C.blueLt],
    ["04", "Deployable End-to-End Web Application", "Flask-based real-time screening web app with interactive handwriting and audio inputs. CPU-friendly (<3s inference, ~85 MB INT8 quantized model). Patient-level stratified CV — no data leakage.", C.teal, C.tealLt],
  ];

  contribs.forEach(([num, title, body, tc, bg], i) => {
    const y = 1.05 + i * 1.05;
    addCard(s, 0.32, y, 9.38, 0.92, bg, tc);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y, w: 0.55, h: 0.92, fill: { color: tc }, line: { color: tc } });
    s.addText(num, { x: 0.32, y: y + 0.28, w: 0.55, h: 0.32, fontSize: 16, color: C.white, bold: true, align: "center" });
    s.addText(title, { x: 0.98, y: y + 0.1, w: 8.55, h: 0.26, fontSize: 13, color: tc, bold: true });
    s.addText(body, { x: 0.98, y: y + 0.38, w: 8.55, h: 0.5, fontSize: 9.5, color: C.textMd });
  });

  // Impact strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y: 5.22, w: 9.38, h: 0.15, fill: { color: C.teal }, line: { color: C.teal } });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 16 — CONCLUSION
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Summary", "Conclusion", 30, "CN");

  // Results recap top
  const rStats = [["93.57%", "Handwriting\n(AUC 0.985)", C.blueLt, C.navy], ["91.70%", "Speech\n(AUC 0.971)", C.tealLt, C.teal], ["96.94%", "CMAFN Fusion\n(AUC 0.9995)", C.navy, C.white]];
  rStats.forEach(([v, l, bg, tc], i) => {
    addCard(s, 0.32 + i * 3.18, 1.0, 3.05, 0.85, bg, bg);
    s.addText(v, { x: 0.42 + i * 3.18, y: 1.04, w: 2.8, h: 0.45, fontSize: 28, color: tc, bold: true, align: "center" });
    s.addText(l, { x: 0.42 + i * 3.18, y: 1.49, w: 2.8, h: 0.32, fontSize: 9.5, color: i === 2 ? "93C5FD" : C.gray, align: "center" });
  });

  const pts = [
    ["Multimodal AI improves screening", "Fusing handwriting and speech via CMAFN yields 96.94% accuracy and 0.9995 AUC-ROC — outperforming both unimodal modules and demonstrating that complementary biomarkers strengthen detection."],
    ["Complementary biomarkers work synergistically", "Handwriting captures fine motor impairment (micrographia, tremor cues), while speech captures voice quality degradation (jitter, shimmer, dysarthria). Together they provide richer diagnostic signals than either alone."],
    ["CMAFN enables SOTA multimodal fusion", "Transformer-based bidirectional cross-attention explicitly models how handwriting and speech features inform each other. GMU gating prevents one modality from dominating, improving over late/early fusion baselines."],
    ["Practical for clinical and remote workflows", "Non-invasive inputs (spiral drawings + short audio) via consumer hardware. Real-time Flask web app with <3s CPU inference. Patient-level stratified 5-fold CV ensures clinically valid performance estimates."],
  ];

  pts.forEach(([h, body], i) => {
    const y = 2.0 + i * 0.8;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y, w: 9.38, h: 0.68, fill: { color: i % 2 === 0 ? C.grayLt : C.white }, line: { color: C.grayMd } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y, w: 0.07, h: 0.68, fill: { color: i % 2 === 0 ? C.navy : C.teal }, line: { color: i % 2 === 0 ? C.navy : C.teal } });
    s.addText(h, { x: 0.48, y: y + 0.07, w: 9.1, h: 0.24, fontSize: 12, color: i % 2 === 0 ? C.navy : C.teal, bold: true });
    s.addText(body, { x: 0.48, y: y + 0.33, w: 9.1, h: 0.32, fontSize: 9.5, color: C.textMd });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 17 — FUTURE WORK
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "Roadmap", "Future Work", 30, "FW");

  const items = [
    ["Extend Biomarkers", "Add gait analysis (IMU sensors), wearable data, and eye-tracking signals as additional non-invasive modalities to further improve screening coverage.", C.navy, C.blueLt],
    ["Longitudinal Monitoring", "Track disease progression over time and support staging beyond binary PD/HC classification — enable early-warning trending using repeated measures.", C.teal, C.tealLt],
    ["Federated Learning", "Multi-center training while keeping patient data local — supports collaboration across hospitals without sharing raw data, enhancing privacy.", C.navy, C.blueLt],
    ["Explainable AI (XAI)", "Integrate attention map visualization and Grad-CAM for clinician trust. Highlight which handwriting/speech features drive predictions.", C.teal, C.tealLt],
    ["Mobile Deployment", "TFLite optimization for on-device real-time screening. Enable low-latency offline inference on mobile devices in remote/rural settings.", C.navy, C.blueLt],
    ["Multilingual Speech", "Evaluate cross-language generalization beyond Italian dataset. Improve robustness for broader demographic applicability worldwide.", C.teal, C.tealLt],
  ];

  items.forEach(([h, body, tc, bg], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.32 + col * 3.18;
    const y = 1.05 + row * 2.2;
    addCard(s, x, y, 3.05, 2.0, bg, tc);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 3.05, h: 0.32, fill: { color: tc }, line: { color: tc } });
    s.addText(`0${i + 1}  ${h}`, { x: x + 0.1, y: y + 0.04, w: 2.85, h: 0.24, fontSize: 10.5, color: C.white, bold: true });
    s.addText(body, { x: x + 0.1, y: y + 0.42, w: 2.85, h: 1.5, fontSize: 9.5, color: C.textMd });
  });

  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 18 — REFERENCES
// =============================================
{
  const s = pres.addSlide();
  addSlideHeader(s, "References", "Selected References", 30, "REF");

  const refs = [
    ["[1]", "Pereira et al., 2018", "\"Handwriting dynamics assessment for early identification of Parkinson's disease,\" Artificial Intelligence in Medicine, 2018.", "Handwriting"],
    ["[2]", "Diaz et al., 2021", "\"A transfer learning approach with MobileNetV2 for Parkinson's disease detection using hand drawings,\" Expert Systems with Applications, 2021.", "Transfer Learning"],
    ["[3]", "Baevski et al., 2020", "\"wav2vec 2.0: A framework for self-supervised learning of speech representations,\" NeurIPS, 2020.", "Self-supervised"],
    ["[4]", "Conneau et al., 2021", "\"Unsupervised cross-lingual representation learning for speech recognition,\" INTERSPEECH, 2021. (XLS-R)", "XLS-R"],
    ["[5]", "Park et al., 2019", "\"SpecAugment: A simple data augmentation method for automatic speech recognition,\" INTERSPEECH, 2019.", "Augmentation"],
  ];

  refs.forEach(([num, auth, title, tag], i) => {
    const y = 1.05 + i * 0.88;
    const bg = i % 2 === 0 ? C.blueLt : C.tealLt;
    const tc = i % 2 === 0 ? C.navy : C.teal;
    addCard(s, 0.32, y, 9.38, 0.78, bg, tc);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.32, y, w: 0.45, h: 0.78, fill: { color: tc }, line: { color: tc } });
    s.addText(num, { x: 0.32, y: y + 0.24, w: 0.45, h: 0.3, fontSize: 12, color: C.white, bold: true, align: "center" });
    s.addShape(pres.shapes.RECTANGLE, { x: 9.28, y: y + 0.12, w: 0.96, h: 0.28, fill: { color: tc }, line: { color: tc } });
    s.addText(tag, { x: 9.28, y: y + 0.12, w: 0.96, h: 0.28, fontSize: 8.5, color: C.white, bold: true, align: "center", valign: "middle" });
    s.addText(auth, { x: 0.88, y: y + 0.06, w: 8.25, h: 0.22, fontSize: 10.5, color: tc, bold: true });
    s.addText(title, { x: 0.88, y: y + 0.3, w: 8.2, h: 0.42, fontSize: 9, color: C.textMd });
  });

  s.addText("Full bibliography available in the IEEE paper. Titles and venues shown for quick attribution during presentation.", { x: 0.32, y: 5.28, w: 9.38, h: 0.18, fontSize: 8.5, color: C.gray, align: "center" });
  s.addText("VIT-AP  •  Senior Design Project Review-2  •  Multimodal Parkinson's Disease Detection", { x: 0.3, y: 5.43, w: 9.4, h: 0.18, fontSize: 7.5, color: C.gray, align: "center" });
}

// =============================================
// SLIDE 19 — THANK YOU
// =============================================
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.5, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.5, w: 10, h: 0.07, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.3, w: 10, h: 0.07, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.37, w: 10, h: 0.255, fill: { color: C.navy }, line: { color: C.navy } });

  s.addText("VIT-AP UNIVERSITY", { x: 0.4, y: 0.1, w: 5, h: 0.32, fontSize: 18, color: C.white, bold: true });
  s.addText("School of Computer Science and Engineering  |  Senior Design Project Review-2", { x: 0.4, y: 0.48, w: 9.2, h: 0.18, fontSize: 10, color: "A5B4FC" });
  s.addText("Multimodal Parkinson's Disease Detection using Cross-Modal Attention Fusion", { x: 0.4, y: 0.72, w: 9.2, h: 0.18, fontSize: 10, color: "93C5FD" });
  s.addText("Dept. of CSE  •  Amaravati, Andhra Pradesh", { x: 0.4, y: 0.96, w: 5, h: 0.18, fontSize: 9, color: "7DD3FC" });

  s.addText("Thank You!", { x: 0.5, y: 1.68, w: 9, h: 0.8, fontSize: 52, color: C.navy, bold: true, fontFace: "Calibri", align: "center" });
  s.addText("Questions? We're happy to discuss methodology, datasets, and deployment details.", { x: 0.5, y: 2.5, w: 9, h: 0.28, fontSize: 13, color: C.gray, align: "center" });

  const kstats = [["96.94%", "Fusion Accuracy"], ["0.9995", "AUC-ROC"], ["5-fold", "Patient-level CV"], ["<3s", "Inference (CPU)"]];
  kstats.forEach(([v, l], i) => {
    addCard(s, 0.8 + i * 2.2, 2.98, 2.0, 0.72, i % 2 === 0 ? C.navy : C.teal, i % 2 === 0 ? C.navy : C.teal);
    s.addText(v, { x: 0.8 + i * 2.2, y: 3.0, w: 2.0, h: 0.38, fontSize: 22, color: C.white, bold: true, align: "center" });
    s.addText(l, { x: 0.8 + i * 2.2, y: 3.38, w: 2.0, h: 0.24, fontSize: 9, color: i % 2 === 0 ? "93C5FD" : "99F6E4", align: "center" });
  });

  // Authors contact
  const authors = [["Tanguturi Venkata Thanuj\n22BCE20003", "thanuj.22bce20003@vitap.ac.in"], ["Katam Krishna Chaitanya\n22BCE7359", "krishna.22bce7359@vitap.ac.in"], ["Pathakuntla Narendar Reddy\n22BCE7707", "narendar.22bce7707@vitap.ac.in"]];
  authors.forEach(([name, email], i) => {
    addCard(s, 0.32 + i * 3.22, 3.88, 3.06, 1.22, C.grayLt, C.grayMd);
    s.addText(name, { x: 0.45 + i * 3.22, y: 3.95, w: 2.8, h: 0.4, fontSize: 10, color: C.navy, bold: true, align: "center" });
    s.addText(email, { x: 0.45 + i * 3.22, y: 4.38, w: 2.8, h: 0.22, fontSize: 8.5, color: C.teal, align: "center" });
  });

  s.addText("Guide: Dr. Rajasekhar Boddu — Asst. Professor, CSE, VIT-AP University", { x: 0.3, y: 5.18, w: 9.4, h: 0.18, fontSize: 9, color: C.navy, align: "center", bold: true });
  s.addText("VIT-AP  •  CSE  |  For screening research use; confirm diagnosis with clinical evaluation.", { x: 0.3, y: 5.4, w: 9.4, h: 0.18, fontSize: 7.5, color: C.white, align: "center" });
}

// Write file
pres.writeFile({ fileName: "/home/claude/PD_Detection_CMAFN.pptx" })
  .then(() => console.log("SUCCESS: PD_Detection_CMAFN.pptx created"))
  .catch(err => console.error("ERROR:", err));
