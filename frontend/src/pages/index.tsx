import { Layout } from '@/components/Layout';
import { ImageUpload } from '@/components/ImageUpload';

export default function Home() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">
            Upload Medical Image for Analysis
          </h2>
          <p className="text-gray-600 mb-6">
            Supported formats: X-ray, CT scan, MRI (.png, .jpg, .jpeg, .dcm, .nii.gz)
          </p>
          <ImageUpload />
        </div>
      </div>
    </Layout>
  );
} 