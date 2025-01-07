# PIXELMAP Framework

**Note! the source code will be available after the paper has been released**

## 📄 **White Paper**
For a detailed explanation of the PIXELMAP framework, including methodology, experiments, and results, refer to our **[White Paper on ArXiv](#)** *(Link coming soon)*.

---

## 🌐 **Interactive Website**
Explore PIXELMAP in action with our **[Interactive Demo](https://photomapping.dogduck.com/)**, where you can upload your own images and visualize the dense correspondence mappings in real-time.

---

**Swarm Intelligence for Dense Image Correspondence**

The **PIXELMAP** framework leverages **swarm intelligence** and **iterative refinement strategies** to establish **dense, robust, and accurate image correspondences**. By conceptualizing each grid cell in an **Affine Correspondence Grid (AC-Grid)** as an autonomous "agent," PIXELMAP iteratively refines local affine transformations and ensures global geometric consistency.

This approach is highly effective for tasks such as:
- **3D Reconstruction**
- **Stereo Matching**
- **Image Stitching**
- **Optical Flow**

---

## 📚 **Overview**
PIXELMAP introduces a novel approach by combining **Correspondence Mapping (CM)** and **Iterative Refinement (IR)**:

- **Correspondence Mapping (CM):** Swarm-inspired dynamics enable local optimizations through agent-based collaboration.
- **Iterative Refinement (IR):** Ensures global consistency by smoothing and refining affine transformations across neighboring grid cells.
- **Affine Correspondence Grid (AC-Grid):** A structured grid representation that balances local adaptability with global coherence.

This synergy results in accurate pixel-to-pixel mappings, even in challenging conditions such as occlusions, geometric distortions, and varying lighting.

---

## 🖼️ **Key Figures**

### Example 1: Monkey Statue Correspondence Mapping
![PIXELMAP applied to two photos of a monkey statue](images/apa_3.png)

### Example 2: Monument Correspondence Mapping
![PIXELMAP applied to two photos of a monument](images/staty_3.png)

### Example 3: Statue 3D Reconstruction
![3D reconstructed with PIXELMAP](images/model3D.png)

### Example 4: Grid Size Comparison on Tree Photos
![Correspondence mapping with varying grid sizes](images/tree_scale.png)

---


## 🚀 **Getting Started**

### **Installation**
```bash
TODO
```

### **Usage**
TODO

---

## 📜 **License**
This project is licensed under the **MIT License** — see the **LICENSE** file for details.

---

For questions or collaborations, feel free to reach out or open an issue.

**Let's redefine image correspondence together!** 🚀
