import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';

export const ImageUpload = () => {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setLoading(true);
      // Show preview
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);

      // Create form data
      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        setPrediction(data.prediction);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.dcm', '.nii.gz']
    },
    multiple: false
  });

  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-blue-500">Drop the medical image here...</p>
        ) : (
          <p>Drag & drop a medical image, or click to select</p>
        )}
      </div>

      {loading && (
        <div className="mt-4 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2">Analyzing image...</p>
        </div>
      )}

      {image && (
        <div className="mt-4">
          <div className="relative h-64 w-full">
            <Image
              src={image}
              alt="Uploaded medical image"
              fill
              className="object-contain rounded-lg"
            />
          </div>
        </div>
      )}

      {prediction && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold mb-2">Analysis Result:</h3>
          <p className="whitespace-pre-wrap">{prediction}</p>
        </div>
      )}
    </div>
  );
}; 