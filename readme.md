## Indic BERT



The repository hosts multilingual ALBERT model trained on Indian languages along with our evolving set of comprehensive general-purpose natural language understanding benchmark, called IGLUE.



#### ALBERT Model





#### IGLUE Benchmark

The IGLUE Benchmark consists of the following tasks:

1. Article Genre Classification
2. Named Entity Recognition
3. Headline Prediction
4. Wikipedia Section Title Prediction
5. Cloze-style Question Answering



We also provide scripts to evaluate performance on publicly available datasets:

* Google TyDi for Bengali and Telugu
* Facebook bAbi 1.2 for Hindi *(publicly available)*
* Amrita University's Paraphrasing task







#### Headline Prediction

* kn: 100k



Example:

```json
    {
        "articleURL": "https://www.kannadigaworld.com/kannada/india-kn/187858.html",
        "content": "Home ಕನ್ನಡ ವಾರ್ತೆಗಳು ರಾಷ್ಟ್ರೀಯ ಯಾಕೂಬ್ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಪಾಲ್ಗೊಂಡವರು ಉಗ್ರರಾಗಿರಬಹುದು: ತ್ರಿಪುರ ರಾಜ್ಯಪಾಲ\nಯಾಕೂಬ್ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಪಾಲ್ಗೊಂಡವರು ಉಗ್ರರಾಗಿರಬಹುದು: ತ್ರಿಪುರ ರಾಜ್ಯಪಾಲ\nPosted By: Karnataka News Bureau Posted date:\nAugust 01, 2015\nIn: ರಾಷ್ಟ್ರೀಯ\nತ್ರಿಪುರ: ಗಲ್ಲು ಶಿಕ್ಷೆಗೆ ಗುರಿಯಾದ ಯಾಕೂಬ್ ಮೆಮನ್ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಪಾಲ್ಗೊಂಡಿದ್ದ ಬಹುತೇಕ ಮಂದಿ ಭಯೋತ್ಪದಕರಾಗಿರಬಹುದು ಎಂದು ಹೇಳುವ ಮೂಲಕ ತ್ರಿಪುರದ ರಾಜ್ಯಪಾಲ ತಥಾಗತ ರಾಯ್ ವಿವಾದಕ್ಕೀಡಾಗಿದ್ದಾರೆ.\n1993 ಮುಂಬೈ ಸರಣಿ ಬಾಂಬ್ ಸ್ಫೋಟದ ಅಪರಾಧಿ ಯಾಕೂಬ್ ಮೆಮನ್ ನ್ನು ನಿನ್ನೆ ಗಲ್ಲಿಗೇರಿಸಿದ ನಂತರ ಮುಂಬೈನಲ್ಲಿ ಅಂತ್ಯಕ್ರಿಯೆ ನಡೆಸಲಾಯಿತು. ಮೆಮನ್ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಭಾಗವಹಿಸದ್ದವರಲ್ಲಿ ಭಯೋತ್ಪಾದಕರೂ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ ಹಾಗಾಗಿ, ಅವರ ಮೇಲೆ ನಿಗಾ ಇಡಿ ಎಂದು ತಥಾಗತ ರಾಯ್ ಹೇಳಿದ್ದರು.\nಗುಪ್ತಚರ ಇಲಾಖೆಯವರು ಮೆಮನ್ ಸಂಬಂಧಿಕರು ಮತ್ತು ಆಪ್ತರನ್ನು ಬಿಟ್ಟ ಬೇರೆ ಯಾರೆಲ್ಲ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಭಾಗವಹಿಸಿದ್ದರೋ ಅಂತಹವರ ಮೇಲೆ ಒಂದು ಕಣ್ಣು ಇಡುವುದು ಒಳ್ಳೆಯದು ಎಂದು ರಾಯ್ ಟ್ವೀಟ್ ಮಾಡಿದ್ದರು.\nರಾಜ್ಯಪಾಲರ ಈ ಹೇಳಿಕೆಗೆ ಟ್ವಿಟರ್ ನಲ್ಲಿ ಟೀಕೆ ಹೆಚ್ಚಾಗುತ್ತಿದ್ದಂತೆ, ಭದ್ರತೆ ದೃಷ್ಟಿಯಿಂದ ನಾನು ಈ ಹೇಳಿಕೆ ನೀಡಿದೆ. ರಾಜ್ಯದ ಭದ್ರತೆ ಬಗ್ಗೆ ಕಾಳಜಿವಹಿಸುವುದು ರಾಜ್ಯಪಾಲರು ಜವಾಬ್ದಾರಿಯಾಗಿರುತ್ತದೆ. ಪೊಲೀಸರು ಯಾಕೂಬ್ ಸಾವಿನ ಶೋಕತಪ್ತರ ಮೇಲೆ ನಿಗಾ ಇಡುವುದರಿಂದ ಮುಂದಾಗವ ಭಯೋತ್ಪಾದನೆಯನ್ನು ತಡೆಯಬಹುದು ಎಂದು ಅಭಿಪ್ರಾಯಪಟ್ಟಿದ್ದರು.\nಇದಕ್ಕೆ, ರಾಜ್ಯಪಾಲರು ಸಮುದಾಯವನ್ನು ಗುರಿಯಾಗಿಟ್ಟುಕೊಂಡು ಈ ಹೇಳಿಕೆ ನೀಡುತ್ತಿದ್ದಾರೆ ಎಂದು ಸಾಮಾಜಿಕ ಜಾಲತಾಣದಲ್ಲಿ ವಿರೋಧಗಳು ವ್ಯಕ್ತವಾದವು. ಮತ್ತೆ ಪ್ರತಿಕ್ರಯಿಸಿದ ರಾಜ್ಯಪಾಲರು, ಇಲ್ಲಿ ನಾನು ಯಾವುದೇ ಸಮುದಾಯದ ಮೇಲೆ ನಿಗಾ ಇಡು ಎಂದು ಹೇಳಿಲ್ಲ. ಹಾಗಾಗಿ, ನನಗೇಕೆ ಅಪರಾಧ ಪ್ರಜ್ಞೆ ಕಾಡಬೇಕು ಎಂದು ಪ್ರಶ್ನಿಸಿದ್ದಾರೆ.\nMost people think this post is Awesome!\nWhat do you think of this post?\nAwesome (1)\n",
        "correctOption": "B",
        "optionA": "ಬಸ್ ನಲ್ಲಿ ಲೈಂಗಿಕ ಕಿರುಕುಳ ಪ್ರಕರಣ: ಬಾಲಕಿ ಸತ್ತಿದ್ದು ದೇವರ ಇಚ್ಛೆ ಎಂದ ಪಂಜಾಬ್ ಸಚಿ",
        "optionB": "ಯಾಕೂಬ್ ಅಂತ್ಯಕ್ರಿಯೆಯಲ್ಲಿ ಪಾಲ್ಗೊಂಡವರು ಉಗ್ರರಾಗಿರಬಹುದು: ತ್ರಿಪುರ ರಾಜ್ಯಪಾ",
        "optionC": "ನಾಪತ್ತೆಯಾದ ಮೀನುಗಾರರನ್ನು ಹುಡುಕಲು ನಾವು ಸಮುದ್ರಕ್ಕೆ ಹಾರಬೇಕೇ?: ಸಚಿವ ನಾಡಗೌ",
        "optionD": "ರಾಜೀನಾಮೆ ಸಲ್ಲಿಸಿದ ಬಳಿಕ ಭಾವನಾತ್ಮಕ ವಿದಾಯ ಭಾಷಣ ಮಾಡಿದ ಯಡಿಯೂರಪ್"
    },

```

