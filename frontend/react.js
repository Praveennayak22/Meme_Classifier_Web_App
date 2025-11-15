import React, { useState } from 'react';
import { Upload, AlertCircle, CheckCircle, Loader, Image, Eye, Brain, MessageSquare, Network, Globe } from 'lucide-react';

const MomentaClassifier = () => {
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('english');

  const languages = [
    { key: 'english', name: 'English', flag: '🇬🇧', desc: 'English only' },
    { key: 'telugu', name: 'Telugu + English', flag: '🇮🇳', desc: 'Code-mixed' },
    { key: 'kannada', name: 'Kannada + English', flag: '🇮🇳', desc: 'Code-mixed' },
    { key: 'tamil', name: 'Tamil + English', flag: '🇮🇳', desc: 'Code-mixed' },
    { key: 'hindi', name: 'Hindi + English', flag: '🇮🇳', desc: 'Code-mixed' }
  ];

  const processingSteps = [
    { name: 'Text Extraction', icon: MessageSquare, desc: 'OCR + IndicBERT' },
    { name: 'Local Features', icon: Eye, desc: 'VGG19 Region Analysis' },
    { name: 'Global Features', icon: Image, desc: 'CLIP Encoding' },
    { name: 'Concept Extraction', icon: Network, desc: 'ConceptNet + Embeddings' },
    { name: 'Classification', icon: Brain, desc: 'MOMENTA Fusion Model' }
  ];

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    setError(null);
    setResult(null);
    setImageFile(file);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const classifyImage = async () => {
    if (!imageFile) return;

    setProcessing(true);
    setCurrentStep(0);
    setError(null);

    try {
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => (prev < processingSteps.length - 1 ? prev + 1 : prev));
      }, 800);

      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('language', selectedLanguage);
      
      const API_URL = 'http://localhost:8000/api/classify';
      
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });

      clearInterval(stepInterval);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      const result = {
        prediction: data.prediction,
        confidence: data.confidence,
        embeddings: data.embeddings,
        extractedText: data.extractedText || 'No text detected',
        concepts: data.concepts || [],
        language: data.language
      };

      setResult(result);
      setCurrentStep(processingSteps.length - 1);
    } catch (err) {
      setError(`Classification failed: ${err.message}. Make sure the backend is running!`);
      console.error('Error:', err);
    } finally {
      setProcessing(false);
      setTimeout(() => setCurrentStep(0), 1000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
             MOMENTA Classifier
          </h1>
          <p className="text-gray-600">
            Multilingual Meme Harmfulness Detection using Multimodal Fusion
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Upload Section */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Upload size={24} className="text-indigo-600" />
              Upload Meme
            </h2>

            {/* Language Selection */}
            <div className="mb-6">
              <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
                <Globe size={18} className="text-indigo-600" />
                Select Language
              </label>
              <div className="grid grid-cols-2 gap-2">
                {languages.map((lang) => (
                  <button
                    key={lang.key}
                    onClick={() => setSelectedLanguage(lang.key)}
                    className={`p-3 rounded-lg border-2 transition-all flex items-center gap-2 ${
                      selectedLanguage === lang.key
                        ? 'border-indigo-600 bg-indigo-50 text-indigo-700'
                        : 'border-gray-200 bg-white hover:border-indigo-300'
                    }`}
                  >
                    <span className="text-xl">{lang.flag}</span>
                    <span className="font-medium text-sm">{lang.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {!preview ? (
              <label
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-3 border-dashed border-indigo-300 rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer hover:border-indigo-500 hover:bg-indigo-50 transition-all"
              >
                <Upload size={48} className="text-indigo-400 mb-4" />
                <p className="text-gray-600 text-center mb-2">
                  Drag & drop an image or click to browse
                </p>
                <p className="text-sm text-gray-400">
                  Supports: JPG, PNG, JPEG
                </p>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border-2 border-gray-200">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full h-auto max-h-96 object-contain"
                  />
                  <button
                    onClick={() => {
                      setPreview(null);
                      setImageFile(null);
                      setResult(null);
                    }}
                    className="absolute top-2 right-2 bg-red-500 text-white px-3 py-1 rounded-lg hover:bg-red-600"
                  >
                    Remove
                  </button>
                </div>

                <button
                  onClick={classifyImage}
                  disabled={processing}
                  className="w-full bg-indigo-600 text-white py-3 rounded-xl font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {processing ? 'Processing...' : `Classify Meme (${languages.find(l => l.key === selectedLanguage)?.name})`}
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="text-red-500 flex-shrink-0" size={20} />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Brain size={24} className="text-purple-600" />
              Analysis Results
            </h2>

            {processing && (
              <div className="space-y-4">
                {processingSteps.map((step, idx) => {
                  const Icon = step.icon;
                  const isActive = idx === currentStep;
                  const isComplete = idx < currentStep;

                  return (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        isActive
                          ? 'border-indigo-500 bg-indigo-50'
                          : isComplete
                          ? 'border-green-300 bg-green-50'
                          : 'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {isComplete ? (
                          <CheckCircle className="text-green-600" size={24} />
                        ) : isActive ? (
                          <Loader className="text-indigo-600 animate-spin" size={24} />
                        ) : (
                          <Icon className="text-gray-400" size={24} />
                        )}
                        <div className="flex-1">
                          <p className="font-semibold text-gray-900">{step.name}</p>
                          <p className="text-sm text-gray-600">{step.desc}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {result && !processing && (
              <div className="space-y-6">
                {/* Main Result */}
                <div
                  className={`p-6 rounded-xl border-2 ${
                    result.prediction === 1
                      ? 'border-red-300 bg-red-50'
                      : 'border-green-300 bg-green-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-2xl font-bold">
                      {result.prediction === 1 ? ' Harmful' : ' Non-Harmful'}
                    </h3>
                    <span className="text-lg font-semibold">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
                    <div
                      className={`h-3 rounded-full ${
                        result.prediction === 1 ? 'bg-red-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                  {result.language && (
                    <div className="flex items-center gap-2 text-sm text-gray-600 mt-2">
                      <Globe size={16} />
                      <span>Analyzed in: <strong>{result.language.name}</strong></span>
                    </div>
                  )}
                </div>

                {/* Embedding Dimensions */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold mb-3 text-gray-700">Feature Dimensions</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(result.embeddings).map(([key, dim]) => (
                      <div key={key} className="bg-white rounded-lg p-3 border border-gray-200">
                        <p className="text-xs text-gray-600 uppercase">{key}</p>
                        <p className="text-lg font-bold text-indigo-600">{dim}d</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Extracted Text */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold mb-2 text-gray-700">Extracted Text</h4>
                  <p className="text-sm text-gray-600 italic break-words">
                    {result.extractedText}
                  </p>
                </div>

                {/* Concepts */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold mb-2 text-gray-700">Detected Concepts</h4>
                  {result.concepts.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {result.concepts.map((concept, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
                        >
                          {concept}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No concepts detected</p>
                  )}
                </div>
              </div>
            )}

            {!processing && !result && (
              <div className="text-center py-12 text-gray-400">
                <Brain size={64} className="mx-auto mb-4 opacity-30" />
                <p>Upload and classify a meme to see results</p>
              </div>
            )}
          </div>
        </div>

        {/* Architecture Info */}
        <div className="mt-8 bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Network size={20} />
            MOMENTA Architecture - Multilingual Support
          </h3>
          <div className="grid md:grid-cols-4 gap-4">
            {[
              { name: 'Text Encoder', desc: 'OCR + IndicBERT (5 Languages)', color: 'bg-blue-50 border-blue-200 text-blue-700' },
              { name: 'Local Features', desc: 'VGG19 + Selective Search', color: 'bg-green-50 border-green-200 text-green-700' },
              { name: 'Global Features', desc: 'CLIP ViT-Base', color: 'bg-purple-50 border-purple-200 text-purple-700' },
              { name: 'Concept Net', desc: 'Caption + Knowledge Graph', color: 'bg-pink-50 border-pink-200 text-pink-700' }
            ].map((module, idx) => (
              <div key={idx} className={`${module.color} border rounded-lg p-4`}>
                <h4 className="font-semibold mb-1">{module.name}</h4>
                <p className="text-sm text-gray-600">{module.desc}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <p className="text-sm text-indigo-700">
              <strong>Supported Languages:</strong> English, Telugu, Kannada, Tamil, Hindi
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MomentaClassifier;
