# Document Classification

## Notebook
[Document Classification Notebook](https://colab.research.google.com/drive/1-LXkOiikkK78nqueEmiKATzH8iH6fXMD?usp=sharing)

Document classification is the process of categorizing scanned or digitized document images into predefined categories based on their content or layout.

- Helpful in cases of MultiPage PDFs
- Improved accuracy
- Faster processing
- Better organization

Here are some key characteristics of the LayoutLMv2 model for document classification:

- It is a transformer-based model that is pre-trained on large datasets of documents to understand document structure and layout. This includes things like sections, paragraphs, tables, lists, etc.

- It uses an additional vector called the 2D position embedding to encode the x and y coordinates of layout elements in the input documents. This allows the model to understand the positional relationships between elements.

- The model achieves state-of-the-art performance on several document image classification benchmarks, outperforming previous layout-aware models like LayoutLM.

- It is particularly effective at classifying documents that have a structured layout with multiple sections, images, titles, etc. The layout encoding helps it understand the hierarchical relationships.

- The self-supervised pre-training allows LayoutLMv2 to generalize to classify new document types without requiring large amounts of labeled training data.

- It incorporates visual structures like paragraphs and tables when making its document classifications, allowing for a holistic interpretation of the document.

- Overall, the layout and vision capabilities allow more accurate classification of real-world documents compared to previous text-only or layout-agnostic methods.

In summary, LayoutLMv2 pushes state-of-the-art performance for document classification while incorporating an understanding of document structure and layout in its predictions.

### Dataset Organization Instructions
1. Convert the PDF into images and organize them into their respective classes.
2. Create a folder named "dataset."
3. Inside "dataset," create two subfolders named "PassiveInvoice" and "CMR."
4. Place the corresponding images into their respective subfolders.

### Dataset Balancing
Make sure to balance the dataset for each of the classes if possible. If balancing is not possible, the `train_test_split` parameter `stratify=data['label']` ensures the same class distribution in both the training and testing sets.

### OCR Engine
`LayoutLMv2Processor` uses PyTesseract, a Python wrapper around Google’s Tesseract OCR engine, under the hood. Note that you can still use your own OCR engine of choice and provide the words and normalized boxes. This requires initializing `LayoutLMv2ImageProcessor` with `apply_ocr` set to False.

For more details, you can find examples [here](https://huggingface.co/spaces/DataIntelligenceTeam/docClassifier1.0).

## Dataset
[Document Classification Dataset](https://drive.google.com/drive/folders/1YoXvXY9qeFMsfzVldIKVvs8TniYfffBh)
