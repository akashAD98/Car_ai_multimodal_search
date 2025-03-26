# 🚗 Car AI Search Engine

A multimodal car search engine powered by LanceDB vector database that enables searching for cars using both text queries and image similarity.

![Car AI Search Engine](https://img.freepik.com/free-vector/modern-blue-urban-adventure-suv-vehicle-illustration_1344-205.jpg)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LanceDB](https://img.shields.io/badge/LanceDB-Latest-green.svg)](https://lancedb.github.io/lancedb/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📋 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Technologies Used](#️-technologies-used)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#️-running-the-application)
- [Project Structure](#️-project-structure)
- [How It Works](#-how-it-works)
- [CSV Format](#-csv-format)
- [Troubleshooting](#️-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

## 🌟 Features

- **Text Search**: Find cars based on natural language queries (e.g., "luxury 7 seater SUV" or "budget car with good mileage")
- **Image Search**: Upload a car image and find similar cars in the database
- **Modern UI**: User-friendly interface with responsive design
- **Vector Search**: Utilizes vector embeddings for semantic understanding of queries
- **Multimodal Capabilities**: Combined text and image search capabilities

## 🎮 Demo

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=Car+AI+Search+Demo" alt="Car AI Search Demo" width="800"/>
  <p><i>Demo of the Car AI Search Engine in action</i></p>
</div>

### Video Demonstration
Check out the video demonstration of the Car AI Search Engine in action. The video shows how to use both text and image search capabilities of the application.

[![Car AI Search Demo Video](https://img.youtube.com/vi/wktA3EB5G8Y/0.jpg)](https://youtu.be/wktA3EB5G8Y?si=XwdpXoW0qeLNKV4i)

*Click on the image above to watch the demo video on YouTube*

## 📊 Dataset

The car image dataset used for this project is available on Google Drive:

[Download Car Image Dataset](https://drive.google.com/drive/folders/1O10sh6VASWpqFWYRyaSBrOKIfU-S060i?usp=sharing)

The dataset contains car images from various manufacturers including:
- Audi
- BMW
- Honda
- Hyundai
- Kia
- Mahindra
- Maruti Suzuki
- Mercedes-Benz
- MG
- Renault
- Tata

Download these images for testing the application's image search capabilities.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/akashAD98/Car_ai_multimodal_search.git
cd Car_ai_multimodal_search

# Install dependencies
pip install -r requirements.txt

# Index your data (run both commands)
python src/indexers/index_text_data.py
python src/indexers/index_image_data.py

# Launch the app
streamlit run src/app.py
```

## 🛠️ Technologies Used

- **LanceDB**: Vector database for storing and searching embeddings
- **Streamlit**: Web application framework for the user interface
- **Sentence Transformers**: For text embeddings (using BAAI/bge-small-en-v1.5)
- **OpenCLIP**: For image embeddings
- **Python**: Core programming language

## 📋 Prerequisites

- Python 3.9 or later
- pip (Python package manager)
- 4GB+ RAM for running the models

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/akashAD98/Car_ai_multimodal_search.git
   cd Car_ai_multimodal_search
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place your car data CSV files in the `data/` directory
   - The CSV should have columns: `car_label`, `car_info`, `car_type`, `fuel_type`, `image_url`

## 🏃‍♂️ Running the Application

### Step 1: Index Your Data

Before running the search engine, you need to index your car data:

1. For text embeddings:
   ```bash
   python src/indexers/index_text_data.py --csv data/text_car_data.csv
   ```

2. For image embeddings:
   ```bash
   python src/indexers/index_image_data.py --csv data/image_car_data.csv
   ```

Optional flags:
- `--cloud`: Use LanceDB Cloud instead of local storage
- `--db_uri`: LanceDB Cloud URI (required if using cloud)
- `--api_key`: LanceDB Cloud API Key (required if using cloud)
- `--region`: LanceDB Cloud region (default: us-east-1)

### Step 2: Launch the Web Interface

Run the Streamlit app:
```bash
streamlit run src/app.py
```

The web interface will automatically open in your default browser.

## 🗂️ Project Structure

```
Car_ai_multimodal_search/
├── data/                   # CSV data files
│   ├── text_car_data.csv
│   └── image_car_data.csv
├── db/                     # Database files (created at runtime)
│   ├── car_ai_text_embeddings/
│   └── car_ai_image_embeddings/
├── src/                    # Source code
│   ├── app.py              # Main application entry point
│   ├── core/               # Core functionality
│   │   └── car_search_core.py
│   ├── indexers/           # Data indexing modules
│   │   ├── index_text_data.py
│   │   └── index_image_data.py
│   └── ui/                 # User interface
│       └── car_search_ui.py
├── temp/                   # Temporary files (created at runtime)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔍 How It Works

### Text Search

1. The app uses Sentence Transformers to convert text queries into embeddings
2. These embeddings are compared against the stored car description embeddings
3. The most semantically similar car descriptions are returned

<div align="center">
  <img src="https://via.placeholder.com/600x300?text=Text+Search+Diagram" alt="Text Search Flow" width="600"/>
</div>

### Image Search

1. User uploads a car image
2. OpenCLIP generates an embedding for the image
3. This embedding is compared against stored car image embeddings
4. The most visually similar cars are returned

<div align="center">
  <img src="https://via.placeholder.com/600x300?text=Image+Search+Diagram" alt="Image Search Flow" width="600"/>
</div>

## 🎯 CSV Format

Your CSV files should be structured as follows:

### Text Car Data CSV:
```
car_label,car_type,fuel_type,car_info,image_url
"Toyota Innova",SUV,Petrol,"A 7-seater SUV with...",https://example.com/innova.jpg
```

### Image Car Data CSV:
```
car_label,car_info,image_url
"Toyota Innova","A 7-seater SUV with...",https://example.com/innova.jpg
```

## ⚠️ Troubleshooting

- **Missing Dependencies**: Ensure you've installed all required packages with `pip install -r requirements.txt`
- **Database Issues**: If search isn't working, check that your indexing steps completed successfully
- **Image Load Failures**: Ensure image URLs are accessible or local paths exist
- **Memory Errors**: Try reducing the batch size if you encounter memory issues during indexing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- **Akash Desai** - [GitHub](https://github.com/akashAD98)

## 🙏 Acknowledgements

- LanceDB for the multimodal vector database capabilities
- Sentence Transformers and OpenCLIP for the embedding models
- Streamlit for the web framework

---

<div align="center">
  <i>This project is for educational purposes and is not affiliated with any car manufacturer.</i>
  
  <p>Repository: <a href="https://github.com/akashAD98/Car_ai_multimodal_search">https://github.com/akashAD98/Car_ai_multimodal_search</a></p>
  <p>Dataset: <a href="https://drive.google.com/drive/folders/1O10sh6VASWpqFWYRyaSBrOKIfU-S060i?usp=sharing">Google Drive - Car Images</a></p>
</div> 
