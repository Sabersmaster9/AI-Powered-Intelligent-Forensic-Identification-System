# AI-Powered Intelligent Forensic Identification System

![Status](https://img.shields.io/badge/Status-In--Development-yellow)
![License](https://img.shields.io/badge/License-Private-red)

An end-to-end automated system designed to standardize and accelerate the judicial expertise process for work-related injury assessments. This project integrates computer vision and Large Language Models (LLMs) to transform raw physical examinations into structured forensic reports.

---

## ⚠️ Important Disclaimer: Data Privacy & Confidentiality
**Please Note:** This repository contains the framework, algorithms, and technical implementation of the system. Due to the sensitive and legal nature of judicial expertise, **all raw judicial documents, specific case data, and confidential forensic reference materials are strictly excluded** from this public repository to comply with privacy regulations and legal ethics.

---

## 🚀 Project Overview
The traditional forensic identification process for work-related injuries often suffers from long processing times, subjective bias, and a heavy clerical burden on forensic doctors. This system provides a digital, contactless, and objective solution through three integrated modules:

### 1. Standard Operating Procedure (SOP) for Machine Vision
The foundation of the system is a specialized SOP tailored for video-based assessment.
* **Standardized Data Acquisition:** Defined protocols for camera positioning, lighting, and patient movement to minimize occlusion.
* **Clinical-Technical Alignment:** Bridges the gap between traditional medical examination requirements and machine vision skeletal tracking needs.
* **Optimized Joint Measurement:** Tailored protocols for different joints to ensure maximal capture accuracy.

### 2. High-Precision Pose Estimation & Biomechanics
Utilizing advanced skeletal tracking to quantify Joint Range of Motion (ROM).
* **Core Engine:** Built on **MMPose**, achieving state-of-the-art accuracy in clinical settings.
* **Key Metrics:** Achieves a **97.11% accuracy rate** with a mean error angle of only **2.41°**, significantly outperforming traditional vision models.
* **Automated Diagnosis:** Instantaneous calculation of biomechanical data, providing objective evidence for disability grading.

### 3. RAG-Enhanced Judicial Document Generation
A specialized Natural Language Processing (NLP) pipeline to automate report writing.
* **Retrieval-Augmented Generation (RAG):** Leverages a local knowledge base of judicial interpretations and forensic standards.
* **Automated Reporting:** Converts the quantitative data from the pose estimation module into standardized, professional forensic report drafts.
* **Efficiency:** Minimizes human error and drastically reduces the time required for administrative documentation.

---

## 🛠 Project Status
This project is currently **under active development**. 
- [x] SOP Initial Drafting & Verification
- [x] Pose Estimation Model Optimization (MMPose Integration)
- [/] RAG Knowledge Base Construction (In Progress)
- [ ] System Integration & GUI Development

## 📄 License
The codebase is provided for research and review purposes. The underlying judicial data remains the property of the respective institutions and is not authorized for public distribution.
