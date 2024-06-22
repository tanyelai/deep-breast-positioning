# Description for labels

## Citation
If you use these labels in your research, please cite as follows:
```
@article{tanyel2024mammographic,
  title={Mammographic Breast Positioning Assessment via Deep Learning},
  author={Tanyel, Toygar and Denizoglu, Nurper and Seker, Mustafa Ege and Alis, Deniz and Karaarslan, Ercan and Aribal, Erkin and Oksuz, Ilkay},
  journal={MICCAI, Deep-Brea3th 2024: A Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care},
  volume={},
  number={},
  pages={},
  year={2024},
  publisher={}
}
```
Will be updated with publication.

## Definition of Findings Used in the Study

| Label                   | Explanation                                                                                   |
|-------------------------|-----------------------------------------------------------------------------------------------|
| Nipple Bounding Box     | A bounding box surrounding the nipple                                                         |
| Pectoralis Muscle Line  | A line delineating and starting from the inferior end of the pectoralis muscle                 |
| Posterior Nipple Line (PNL) | A line perpendicular to the pectoralis muscle line starting from the center of the nipple bounding box |
| Qualitative Quality     | An image-level quality label given by an expert radiologist based on PNL criteria and further practical experience |

## Label Description

In the labeling process, we used the publicly available VinDr Mammography open-access large-scale dataset (Nguyen et al., 2023). The VinDr Mammography dataset includes 5000 mammography exams collected from opportunistic screening settings in two hospitals in Vietnam between 2018 and 2020.

For our labeling, we randomly selected 1000 exams from the VinDr Mammography dataset, including only mediolateral oblique (MLO) views. 

### Quantitative Labels (data)
Ground truth annotations were carried out by two board-certified breast radiologist (N.D. and E.C.) with over five years of experience in breast imaging. The radiologist annotated the mammograms using a specialized workstation equipped with a browser-based annotation tool (https://matrix.md.ai) and a 6-megapixel diagnostic monitor (Radiforce RX 660, EIZO). All mammograms were reviewed in the Digital Imaging and Communications in Medicine (DICOM) format.

The radiologist (N.D.) annotated the nipple and the pectoralis muscle line from the inferior end on MLO views, following the guidelines of the American College of Radiology and the Royal Australian and New Zealand College of Radiologists (Australian Screening Advisory Committee, 2001; Hendrick et al., 1999; Royal Australian and New Zealand College of Radiologists, 2002).

### Qualitative Labels (qualitativeLabel)
After the lines and nipple bounding box other radiologist (E.C.) assessed the images for breast positioning and classified the MLO views as poor or good based on ACR quality standards (Hendrick et al., 1999).

> **Note:**
> We do not explicitly provide coordinates for PNL; however, the PNL can be automatically generated using a 90° rule, extending from the nipple coordinate to the pectoral muscle line. This ensures that all PNLs are perpendicular to the pectoral muscle during the evaluation phase.

<img width="450" alt="image" src="https://github.com/tanyelai/deep-breast-positioning/assets/44132720/e4c116b0-df16-4092-aea4-cee7209c7263">

### Dataset Details

- **StudyInstanceUID**: Unique identifier for the study (exam) instance.
- **SOPInstanceUID**: Unique identifier for the image instance of an exam.
- **annotationMode**: The mode of annotation used (e.g., "line" or "bbox").
- **labelName**: Specific label names like Pectoralis and Nipple.
- **data**: Contains coordinates for vertices for lines and for nipples.
- **qualitativeLabel**: An image-level qualitative label provided by another expert radiologist.
- **height**: The height of the image in pixels.
- **width**: The width of the image in pixels.
- **SeriesDescription**: Images’ view information (e.g., L-MLO or R-MLO).
- **ImagerPixelSpacing**: Pixel spacing of the image.
- **SeriesInstanceUID**: Unique identifier for the series instance.
- **ManufacturerModelName**: The manufacturer's model name of the imaging device.
- **PhotometricInterpretation**: The photometric interpretation used in the image.
- **Split**: Designation of the dataset split including Train, Validation, and Test.

### References

1. Australian Screening Advisory Committee, 2001. National Accreditation Standards BreastScreen Australia Quality Improvement Program (Revised).
2. Hendrick, R.E., Bassett, L., Botsco, M.A., Deibel, D., Feig, S., Gray, J., Haus, A., Heinlein, R., Kitts, E.L., McCrohan, J., Monsees, B., 1999. Mammography quality control manual. Royal American College of Radiologists.
3. Nguyen, H.T., Nguyen, H.Q., Pham, H.H., Lam, K., Le, L.T., Dao, M., Vu, V., 2023. VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography. Sci Data 10, 277. [DOI](https://doi.org/10.1038/s41597-023-02100-7)
4. Royal Australian and New Zealand College of Radiologists, 2002. Mammography quality assurance program.