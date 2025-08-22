# Conjoint Analysis D-Efficiency Tool

A comprehensive Streamlit web application for designing and optimizing conjoint experiments for market research and demand estimation analysis.

## 🎯 Features

- **Interactive Design Setup**: Define product attributes and levels through an intuitive interface
- **D-Efficiency Optimization**: Automatically calculate and optimize design efficiency
- **Visual Analytics**: Interactive charts showing efficiency curves and trade-offs
- **Cost-Benefit Analysis**: Evaluate survey costs vs. statistical precision
- **Export Capabilities**: Download results in Excel, CSV, and JSON formats
- **Configuration Import/Export**: Save and share experiment designs

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
```bash
git clone <your-repo-url>
cd conjoint-analysis-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run conjoint_app.py
```

4. **Open your browser** to `http://localhost:8501`

## 📖 How to Use

### 1. Design Setup
- Navigate to the "🔧 Design Setup" page
- Define your product attributes (e.g., Price, Brand, Features)
- Specify levels for each attribute
- Set study parameters (respondents, alternatives, target efficiency)

### 2. Analysis
- Go to "📊 Analysis" to run the optimization
- View the D-efficiency curve
- Identify the optimal number of questions per respondent

### 3. Results
- Review detailed insights and recommendations
- Analyze cost-benefit trade-offs
- Get survey logistics estimates

### 4. Export
- Download comprehensive Excel reports
- Export configuration files for future use
- Import previously saved configurations

## 🧮 Understanding D-Efficiency

D-efficiency measures how well a conjoint design can estimate main effects and interactions:

- **1.0**: Perfect efficiency (full factorial design)
- **0.8+**: Generally considered good efficiency
- **0.6-0.8**: Acceptable for many applications
- **<0.6**: May require more respondents or questions

The tool automatically calculates the minimum number of questions needed to achieve your target efficiency.

## 📊 Example Use Cases

### Market Research
- Test consumer preferences for new product features
- Optimize pricing strategies
- Evaluate brand positioning

### Product Development
- Prioritize feature development
- Understand feature trade-offs
- Guide design decisions

### Operations Research
- Optimize service offerings
- Evaluate process improvements
- Resource allocation decisions

## 🔧 Technical Details

### Core Algorithm
The application implements a simplified D-efficiency calculation based on:
1. Full factorial design generation
2. Fractional factorial sampling when needed
3. Information matrix calculation (X'X)
4. Determinant-based efficiency computation

### Key Components
- **ConjointDesigner Class**: Core optimization logic
- **Streamlit Interface**: Interactive web application
- **Plotly Visualizations**: Dynamic charts and graphs
- **Export Functions**: Multi-format data export

## 📁 File Structure

```
conjoint-analysis-tool/
├── conjoint_app.py          # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── examples/               # Example configurations (optional)
```

## 🌐 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### Heroku
1. Add `Procfile`: `web: streamlit run conjoint_app.py --server.port $PORT --server.enableCORS false`
2. Deploy using Heroku CLI or GitHub integration

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY conjoint_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "conjoint_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔬 Advanced Features

### Custom Efficiency Functions
Modify the `calculate_d_efficiency` method to implement:
- A-efficiency optimization
- Custom information matrix calculations
- Bayesian D-efficiency with priors

### Integration Options
- Connect to survey platforms (Qualtrics, SurveyMonkey)
- Export to R for advanced analysis
- Integration with choice modeling packages

## ⚠️ Limitations

- Simplified D-efficiency calculation (suitable for most applications)
- Maximum recommended attributes: 10
- Optimal for 2-5 levels per attribute
- Processing time increases with design complexity

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all packages are installed
pip install -r requirements.txt --upgrade
```

**Memory Issues with Large Designs**
- Reduce number of attributes or levels
- Increase minimum questions parameter
- Consider fractional factorial approaches

**Slow Performance**
- Reduce the range of questions tested
- Use fewer respondents for initial testing
- Simplify attribute structure

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Streamlit documentation
3. Create an issue in the repository

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Based on R's `cbcTools` and `idefix` packages
- Inspired by conjoint analysis methodologies
- Built with Streamlit, Plotly, and pandas
- D-efficiency calculations adapted from experimental design literature

## 🔮 Future Enhancements

- [ ] Advanced efficiency criteria (A-optimal, I-optimal)
- [ ] Bayesian D-efficiency with prior information
- [ ] Integration with choice modeling analysis
- [ ] Support for alternative-specific attributes
- [ ] Blocked experimental designs
- [ ] Real-time collaboration features