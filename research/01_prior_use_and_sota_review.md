# Prior use of the HUSJA-UA dataset and state of the art in hip fracture X-ray AI

Prepared 26 Nov 2026 for the Segmentation Project. Research only: no idea is chosen here.

---

## Section 1. Bottom line

1. **No published paper, preprint, GitHub repository or Kaggle notebook uses the HUSJA-UA femoral fracture dataset besides the original authors.** The dataset went public on IEEE DataPort on 11 Nov 2025, so it is only about one year old.
2. **The base article (González et al. 2025) has 25 citing works in OpenAlex (Springer shows 23).** I checked every one by title, and read the full text of the 8 that could plausibly use the data. All of them cite it only as related work, and all use private or other public datasets.
3. **This means a paper on HUSJA-UA would be among the first, and possibly the first, independent benchmarks on it.** This is a real selling point for a Q2/Q3 venue, as long as the method also brings something new.
4. The open gaps that fit this dataset best are covered in Section 6. In short: rare-class imbalance (Garden I/II neck and subtrochanteric), segmentation quality (Dice only 0.77, no per-class report), joint segmentation plus fine-grained grading, label-efficient learning, uncertainty and calibration, and cross-dataset testing.

---

## Section 2. How the search was done (step by step)

| Step | What I searched | Result |
|---|---|---|
| 1 | IEEE DataPort record of the dataset, including comments and linked papers | Only the base article is linked. One user comment. |
| 2 | Citation index (OpenAlex API, "works citing" the base DOI) | 25 citing works, listed in Section 4 |
| 3 | Full-text search in OpenAlex for "HUSJA" | 0 relevant hits (only unrelated non-English works) |
| 4 | Web search for the dataset DOI `10.21227/xny1-zz29` | 0 papers citing the dataset DOI |
| 5 | GitHub repository and code search for "HUSJA" and "femur_HUSJA_UA" | 0 relevant repositories |
| 6 | Kaggle: notebooks on your uploaded copy (utopianstar/dataset) plus a general Kaggle search | Your copy has 0 notebooks; no other copy found |
| 7 | Web search for 2023–2026 hip/femoral fracture detection, classification and segmentation | Key papers summarised in Section 5 |

Limits of the search: Google Scholar and Scopus were not queried directly (they block automated access). Springer's own count (23) and OpenAlex (25) agree closely, so the list is near complete. A few citing papers are paywalled; for those I read the abstract and the parts that were open.

---

## Section 3. What we know about the dataset itself

From the [IEEE DataPort record](https://ieee-dataport.org/documents/husja-ua-femoral-fracture) (DOI 10.21227/xny1-zz29) and the [base article](https://insightsimaging.springeropen.com/articles/10.1186/s13244-025-01958-y):

- **Source:** Hospital Universitario de San Juan de Alicante, Spain, 2010–2021. 10,300 PACS images from 2,609 patients, of which 986 studies (707 patients) were annotated.
- **Unit of data:** cropped femur ROIs (100% margin), 8-bit. 1,454 femurs after exclusions.
- **Annotation:** polygon masks of the fracture region by two radiologists (3 and 20 years of experience); test set reviewed by a musculoskeletal committee. YOLO-seg format, label map in `data.yml`.
- **Licence:** CC BY 4.0, research use. Download 798.68 MB (`femur_HUSJA_UA.zip`).
- **Patient-level split and class counts (femurs):**

| Class | Total | Train | Val | Test |
|---|---|---|---|---|
| No fracture | 495 | 363 | 67 | 65 |
| Neck, Garden I/II (undisplaced) | 84 | 57 | 11 | 16 |
| Neck, Garden III/IV (displaced) | 337 | 254 | 42 | 41 |
| Pertrochanteric, Evans I/II | 205 | 134 | 37 | 34 |
| Pertrochanteric, Evans III/IV/V | 275 | 199 | 34 | 42 |
| Subtrochanteric | 58 | 36 | 10 | 12 |
| **Total** | **1,454** | **1,043** | **201** | **210** |

**Warning to check in EDA:** a user commented on DataPort on 22 Sep 2026: "Why is there nothing inside the val folder?" No reply from the authors. We must check whether our `data/val` folder is empty. If it is, we will need to rebuild a patient-level validation split from train, and say so in the paper.

**Base article results to beat** (YOLOv8-small-seg, 1024 px, six runs per setting):
- Fracture vs no fracture: AUC 0.981, accuracy 96.2%, sensitivity 95.2%, specificity 98.5%.
- Six-class type accuracy: 86.19%.
- Segmentation: mean Dice 0.77 (95% CI 0.56–0.98). **No per-class Dice, IoU or mask mAP reported.**
- CNN classification baselines (VGG16, DenseNet121, EfficientNetB4, InceptionV3, ResNet50V2): accuracy 74–84%.

**Weak points the authors state themselves:**
- Small annotated set; few undisplaced neck and subtrochanteric fractures, and the model is worst on those two.
- Single institution (although 5 devices, DX and CR).
- No clinical reader study yet.
- Future work: semi-automatic labelling to enrich rare fractures; testing whether segmentation helps more than a box or arrow; formal generalisation study.

---

## Section 4. Every work that cites the base article (25), and whether it used HUSJA-UA

None used HUSJA-UA. "Read" means I checked the full text or the open parts.

### 4a. Closest competitors (hip/femur, read in detail)

| # | Paper | Data | Task and method | Key numbers | Uses HUSJA-UA? |
|---|---|---|---|---|---|
| 1 | Ahmed et al., [Beyond prompt-only segmentation: foundation model–guided trust-region label refinement for single-stage bone fracture instance segmentation](https://doi.org/10.1016/j.engappai.2026.115849), *Eng Appl AI* 2026 (Q1) | FracAtlas (4,083 studies, 338 hip) + Kaggle/Roboflow upper limb | MedSAM refines ground-truth masks at training time (IoU gate, boundary band); YOLOv8/9/11-seg, vs Mask R-CNN, Mask DINO, Mask2Former; new skeleton-aware metric | mAP50 0.63 on held-out test | No. Cites González only as "Dice 0.77" prior work |
| 2 | Shen et al., [Improved YOLOv8 with ADown for intertrochanteric fractures (AO/OTA)](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1759383/full), *Front Med* 2026 | 976 private images, one hospital | YOLOv8n detection with average-pool downsampling; A1/A2/A3 | mAP50 0.817 vs 0.805 baseline | No |
| 3 | Zhao et al., [YOLOv8 with Hybrid Attention Gating Block for intertrochanteric fracture detection](https://doi.org/10.1007/s10278-026-02230-5), *J Imaging Inform Med* 2026 | 694 private pelvic radiographs | YOLOv8s + attention gating; repeated 5-fold CV | Precision 0.893, mAP50 0.866 | No |
| 4 | Mohaidat et al., [Hybrid PEFT-ViT and XGBoost for hip fracture classification](https://link.springer.com/article/10.1007/s10278-026-02395-z), *J Imaging Inform Med* 2026 | 768 images, KAUH Jordan (Kaggle release promised) | LoRA-tuned ViT-B/16 features into XGBoost; 3 classes | Accuracy 88.3% on 77 test images | No |
| 5 | Nie et al., [TRFC-net](https://qims.amegroups.org/article/view/156198/html), *QIMS* 2026 | 1,238 private images | RetinaNet crop, then Swin-T screening and A1/A2/A3 typing, Grad-CAM, reader study | Accuracy 0.902; residents improved 0.817 to 0.924 | No. States no public trochanteric dataset exists |
| 6 | Yang et al., [Garden classification of femoral neck fracture using deep learning](https://www.nature.com/articles/s41598-025-27766-4), *Sci Rep* 2025 | 1,588 private patients, AP + lateral, CT as reference | Hip-joint detection then 12 classifiers for Garden I–IV | Accuracy 90.6% | No |

### 4b. Other citing works (checked by title and venue; none are HUSJA-UA studies)

- Reviews and meta-analyses: AI for facial fracture detection (*Oral Radiol* 2025); AI in orthopaedics bibliometrics (*J Multidiscip Healthc* 2025); AI in emergency musculoskeletal imaging (*Diagn Interv Imaging* 2026); AI for traumatic femur fracture, meta-analysis (*Surgery* 2026); YOLO models for fracture detection, meta-analysis ([JIIM 2026](https://link.springer.com/article/10.1007/s10278-026-02341-z)); sustainable AI in imaging (*Digital Health* 2026); Portuguese fracture AI review (*Revista DCS* 2026).
- External validation of commercial tools: hip and pelvic fracture tool, multicentre (*Eur J Radiol* 2026); hip fracture AI real-world test (*Medical Records* 2026).
- General fracture models on mixed public datasets: multi-channel fusion ResNet (*IET Image Process* 2026); YOLOv11 bone fracture (IEEE conf. 2025); FractureNet Swin (IEEE conf. 2025); CNN + MobileNetV2 (LNEE 2026); two-stage YOLOv8 pipeline (LNNS 2026); YOLOv8 severity quantification (Springer AIS 2026); DNN vs CNN (*IJATIS* 2026).
- Hip-specific, private data: Vietnamese study on femoral neck and intertrochanteric fracture detection (*Tạp chí Nghiên cứu Y học* 2026).
- Unrelated (citation noise): knee osteoarthritis synovium segmentation; EV battery review.

**What this tells us:** the field is citing González et al. as the reference for "segmentation-based hip fracture AI", but nobody has yet built on their public data. Most competitors are small private single-centre studies that tune YOLO with one attention block, with gains of about 1 point mAP and no variance reported. Reviewers are getting tired of that pattern, so our method needs a clearer idea than "YOLO + new module".

---

## Section 5. State of the art, 2023–2026 (beyond the citing works)

### 5a. Detection and classification (strongest numbers)

| Paper | Data | Method | Numbers | Note |
|---|---|---|---|---|
| OccuNet, [*Radiology* 2026](https://pubs.rsna.org/doi/10.1148/radiol.253931) | 2,576 adults, 4 hospitals, CT/MRI reference | Contrastive pretraining on original vs artifact-augmented radiographs, then scale-adaptive detector | Sensitivity 97.5%, specificity 98.8%, AUC 0.99; occult cases 94.7% vs radiologists 86.2% | Top-tier; focuses on occult neck fractures |
| Wang et al., [*Bioengineering* 2025](https://mdpi.com/2306-5354/12/5/466) | 2,032 images, 2 hospitals + external 344 | ResNet34 with femoral-neck segmentation guidance and concept encoder | Accuracy 94.8%, AUC 0.991; external 94.0% | Undisplaced (Garden I/II) neck only |
| Yıldız Potter et al., [*Sci Rep* 2024](https://www.nature.com/articles/s41598-024-63001-2) | 823 private + FracAtlas; PelvixNet external | VarifocalNet FPN | AUC 0.98, localisation IoU 0.67–0.71 | Box-level only |
| Bhuiyan et al., [*Sci Rep* 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13451365/) | PXR150 (public) + 228 private | 9-bone segmentation (U-Net, MiT-B0) then ROI fracture classifier | Bone Dice 0.97; fracture AUC 0.82–0.84 | Anatomy segmentation, not fracture-region segmentation |

### 5b. Segmentation of the fracture region on hip X-rays

This is the thinnest part of the literature. Almost all hip work is classification or boxes. Fracture-region masks on radiographs exist only in:
- González et al. 2025 (HUSJA-UA, Dice 0.77, YOLOv8-seg).
- FracAtlas-based work (mixed anatomy, only 63 fractured hip studies), including Ahmed et al. 2026 (mAP50 0.63).

Foundation models (SAM, MedSAM, SAM2) are being tested for bone segmentation, but mostly on CT or for anatomy ([Magg et al., arXiv 2026](https://arxiv.org/pdf/2603.10541)). Their use for **fracture-line or fracture-region masks on radiographs** is barely explored, apart from Ahmed et al. using MedSAM only to clean labels.

### 5c. Public datasets we could use for extra testing

- [FracAtlas](https://www.nature.com/articles/s41597-023-02432-4): masks, hip subset is small (338 studies, 63 fractured). Usable as a zero-shot transfer test.
- PXR150 (pelvic, image-level labels) and PelvixNet (external, image-level): usable only for detection-level transfer.
- No other public hip dataset has fracture-region masks with Garden/Evans grades. **HUSJA-UA is currently unique in that respect**, which is useful to state in the paper.

---

## Section 6. Open gaps that look feasible for a Q2/Q3 paper

Each gap is linked to evidence from Sections 3–5. I have not ranked or picked one; that is the next step.

1. **Rare-class failure.** Garden I/II neck (57 train femurs) and subtrochanteric (36) are the classes the base model gets wrong. No one has tried long-tail methods on this data (class-balanced or focal losses, decoupled heads, targeted copy-paste of fracture masks, diffusion or generative augmentation of rare classes).
2. **Weak and unreported segmentation quality.** Mean Dice 0.77 with a wide CI (0.56–0.98) and no per-class Dice, IoU, Hausdorff or boundary metrics. A dedicated segmentation architecture (for example a hybrid CNN–transformer or boundary-aware decoder) with a full per-class report would be a clear step up.
3. **Classification and segmentation in one network, done better.** The base work reads the class off YOLO-seg masks. A multi-task model where the mask guides fine-grained grading (Garden or Evans), or a hierarchical head (location first, then grade), matches how radiologists reason and matches the 15 "same location, wrong grade" errors the authors report.
4. **Label-efficient learning, which is their own stated future work.** About 9,300 unannotated PACS images exist in the source hospital, but they are not public. With only the public data, semi-supervised or self-supervised pretraining could use FracAtlas or other unlabelled radiographs, or SAM/MedSAM pseudo-masks. Ahmed et al. 2026 showed naive SAM pseudo-labels hurt, which leaves room for a better refinement scheme.
5. **Foundation models for fracture-region segmentation.** Fine-tuning SAM/MedSAM/SAM2 with adapters (LoRA or similar) on HUSJA-UA, or a prompt-free variant, has not been reported for hip fractures.
6. **Uncertainty, calibration and interpretability.** The paper's title promises interpretability, but there is no calibration, uncertainty map or failure analysis. Mask-level uncertainty (for example Monte Carlo dropout or ensembles) to flag low-confidence cases for the radiologist is open.
7. **Generalisation.** Single centre. A cross-dataset test on FracAtlas hip images (zero-shot or few-shot) would address the authors' stated limitation, even if only at detection level.
8. **Reproducible benchmark.** Because nobody else has published on HUSJA-UA, a careful benchmark of modern models (YOLOv8/11-seg, Mask2Former, nnU-Net, SAM-based) with repeated runs and per-class metrics is itself a contribution. On its own it is probably Q3/Q4; combined with one new method it fits Q2/Q3.

**Feasibility notes for Kaggle:** 1,043 training femurs, 1024 px crops. Training YOLO-seg, a U-Net-family model or a LoRA-tuned SAM on a Kaggle P100/T4 is realistic within the 12-hour session limit. Six-run repeats (as in the base article) are affordable for medium models.

---

## Section 7. Sources

- Dataset: [IEEE DataPort, HUSJA UA Femoral Fracture](https://ieee-dataport.org/documents/husja-ua-femoral-fracture)
- Base article: [Insights into Imaging](https://insightsimaging.springeropen.com/articles/10.1186/s13244-025-01958-y), [PubMed 40232323](https://pubmed.ncbi.nlm.nih.gov/40232323/), [PMC12000489](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12000489/)
- Citing works list: OpenAlex API, works citing W4409448996 (queried 26 Nov 2026)
- All other papers are linked in Sections 4 and 5.
