import Link from "next/link"
import { ArrowUpCircle, BookOpen, Database, FlaskRoundIcon as Flask, LineChart, ScrollText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export default function MLProposal() {
  // Smooth scroll function
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    element?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 hidden md:flex">
            <Button variant="ghost" className="mr-2" onClick={() => scrollToSection("intro")}>
              <BookOpen className="mr-2 h-4 w-4" />
              Introduction
            </Button>
            <Button variant="ghost" className="mr-2" onClick={() => scrollToSection("problem")}>
              <Flask className="mr-2 h-4 w-4" />
              Problem
            </Button>
            <Button variant="ghost" className="mr-2" onClick={() => scrollToSection("methods")}>
              <Database className="mr-2 h-4 w-4" />
              Methods
            </Button>
            <Button variant="ghost" className="mr-2" onClick={() => scrollToSection("results")}>
              <LineChart className="mr-2 h-4 w-4" />
              Results
            </Button>
            <Button variant="ghost" onClick={() => scrollToSection("references")}>
              <ScrollText className="mr-2 h-4 w-4" />
              References
            </Button>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-6">
        {/* Title Section */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-bold tracking-tight">ML Project Proposal</h1>
          <p className="mt-4 text-lg text-muted-foreground">CS 4641 Group 58 • Georgia Institute of Technology • Nannan Aravazhi, Pranavkrishna Suresh, Henry Lin, Gabriel Ferrer, Mihir Balsara</p>
        </div>

        {/* Introduction Section */}
<section id="intro" className="mb-16 scroll-mt-16">
  <Card className="p-6">
    <h2 className="text-2xl font-bold mb-4">Introduction & Background</h2>
    <div className="space-y-6">
      
      {/* Literature Review */}
      <div>
        <h3 className="text-xl font-semibold mb-2">Literature Review</h3>
        <p className="text-muted-foreground">
          Deep learning architectures play a crucial role in image classification and feature extraction:
        </p>
        <ul className="list-disc list-inside text-muted-foreground mt-2 space-y-2">
          <li>
            <strong>VGG-19:</strong> Utilizes stacked 3×3 convolutions, max pooling, and fully connected layers for feature extraction. While effective, it has high computation costs and inefficient gradient flow [1].
          </li>
          <li>
            <strong>ResNet-50:</strong> Addresses vanishing gradients with residual connections, enabling deeper networks and stable convergence [2].
          </li>
          <li>
            <strong>DenseNet-121:</strong> Uses densely connected layers to enhance gradient propagation and reduce redundancy, preserving low-dimensional features for improved performance in data-scarce environments [3].
          </li>
        </ul>
      </div>

      {/* Dataset Description */}
      <div>
        <h3 className="text-xl font-semibold mb-2">Dataset Description</h3>
        <p className="text-muted-foreground">
          The following models are trained on ImageNet, leveraging large-scale labeled datasets for feature learning:
        </p>
        <ul className="list-disc list-inside text-muted-foreground mt-2 space-y-2">
          <li>
            <strong>VGG-19:</strong> Trained on ImageNet (1.2M+ labeled images, 1,000 categories), making it a benchmark for deep learning. Supports transfer learning and feature extraction across various applications.
          </li>
          <li>
            <strong>ResNet-50:</strong> Learns hierarchical features using skip connections, improving convergence and mitigating vanishing gradients. Suitable for fine-tuning in detection and segmentation.
          </li>
          <li>
            <strong>DenseNet-121:</strong> Uses densely connected layers for feature reuse, enhancing learning efficiency. Its robust feature extraction benefits image classification and medical analysis.
          </li>
        </ul>
      </div>

      {/* Dataset Links */}
      <div>
        <h3 className="text-xl font-semibold mb-2">Dataset Links</h3>
        <ul className="list-disc list-inside text-primary mt-2 space-y-2">
          <li>
            <Link href="https://www.kaggle.com/datasets/crawford/vgg19" className="hover:underline">
              VGG-19 Dataset
            </Link>
          </li>
          <li>
            <Link href="https://www.kaggle.com/datasets/crawford/resnet50" className="hover:underline">
              ResNet-50 Dataset
            </Link>
          </li>
          <li>
            <Link href="https://www.kaggle.com/datasets/pytorch/densenet121" className="hover:underline">
              DenseNet-121 Dataset
            </Link>
          </li>
        </ul>
      </div>

    </div>
  </Card>
</section>


        {/* Problem Definition Section */}
        <section id="problem" className="mb-16 scroll-mt-16">
          <Card className="p-6">
            <h2 className="text-2xl font-bold mb-4">Problem Definition</h2>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-semibold mb-2">Problem Statement</h3>
                <p className="text-muted-foreground">Distracted driving is among the prevalent causes of road accidents, often resulting in severe injuries and fatalities. Current surveillance measures rely on manual enforcement or post-incident analysis, with no real time intervention.
This project intends to use machine learning to detect distracted driving in real time. This will encourage driver awareness, improve vehicle safety, and prevent accidents.
</p>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Motivation</h3>
                <p className="text-muted-foreground">With the widespread use of smartphones and other in-car distractions, automated systems for detection of risky driving behaviors are important. Traditional enforcement methods are limited in real-time effectiveness. An AI-driven solution is capable of detecting distractions in real time, aiding driver assist systems, insurance assessments, and regulatory enforcement. By reducing road accidents, this solution can ultimately help save lives.</p>
              </div>
            </div>
          </Card>
        </section>

        {/* Methods Section */}
        <section id="methods" className="mb-16 scroll-mt-16">
          <Card className="p-6">
            <h2 className="text-2xl font-bold mb-4">Methods</h2>
            <div className="space-y-6">
              
              {/* Data Preprocessing Methods */}
              <div>
                <h3 className="text-xl font-semibold mb-3">Data Preprocessing Methods</h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>
                    <strong>Image Preprocessing with OpenCV (cv2):</strong> Standardizes image sizes and prepares them for deep learning models.
                  </li>
                  <li>
                    <strong>Image Loading, Reshaping, and Flattening:</strong> Images are stored as NumPy arrays and can be flattened for classical machine learning.
                  </li>
                  <li>
                    <strong>Data Augmentation with imgaug.augmenters:</strong> Enhances training data using transformations such as:
                    <ul className="list-disc list-inside ml-6">
                      <li>Rotation</li>
                      <li>Shear</li>
                      <li>Flipping</li>
                      <li>Scaling</li>
                    </ul>
                    Improves generalization and reduces overfitting.
                  </li>
                  <li>
                    <strong>Feature Engineering with KerasClassifier:</strong> Integrated into a scikit-learn pipeline for combining deep learning with classical machine learning. Supports feature extraction using models like VGG16 and ResNet50.
                  </li>
                </ul>
              </div>

              {/* ML Algorithms/Models */}
              <div>
                <h3 className="text-xl font-semibold mb-3">ML Algorithms/Models</h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>
                    <strong>Deep Learning Models:</strong>
                    <ul className="list-disc list-inside ml-6">
                      <li>
                        <strong>Transfer Learning with ResNet50 & VGG16:</strong> Uses pretrained convolutional bases with additional dense layers for classification, leveraging prior knowledge from ImageNet.
                      </li>
                      <li>
                        <strong>CNN:</strong> Extracts spatial features through convolutional layers, pooling, and dense layers. Captures spatial hierarchies to help distinguish driver behaviors.
                      </li>
                      <li>
                        <strong>Dense Neural Network:</strong> Uses flattened image input followed by dense layers with ReLU and dropout for non-linear learning and overfitting reduction.
                      </li>
                    </ul>
                  </li>
                  <li>
                    <strong>Classical Models:</strong>
                    <ul className="list-disc list-inside ml-6">
                      <li><strong>K-Nearest Neighbors:</strong> Baseline classifier using nearest neighbors, effective for balanced datasets.</li>
                      <li><strong>Logistic Regression:</strong> Computes class probabilities efficiently.</li>
                    </ul>
                  </li>
                </ul>
              </div>

              {/* Learning Methods */}
              <div>
                <h3 className="text-xl font-semibold mb-3">Learning Methods</h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>
                    <strong>Supervised Learning:</strong> All discussed methods fall under supervised learning, where images are labeled to enable direct class mapping, making it effective for classification tasks.
                  </li>
                </ul>
              </div>

            </div>
          </Card>
        </section>

        {/* Results Section */}
        <section id="results" className="mb-16 scroll-mt-16">
          <Card className="p-6">
            <h2 className="text-2xl font-bold mb-4">Potential Results & Discussion</h2>
            <div className="space-y-6">
              
              {/* Quantitative Metrics */}
              <div>
                <h3 className="text-xl font-semibold mb-3">Quantitative Metrics</h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>
                    <strong>Accuracy:</strong> Measures overall correctness across all classes.
                  </li>
                  <li>
                    <strong>Precision:</strong> Evaluates true positive predictions among all predicted positives.
                  </li>
                  <li>
                    <strong>Recall:</strong> Assesses the model’s ability to detect relevant instances.
                  </li>
                  <li>
                    <strong>F1 Score:</strong> Balances precision and recall for overall effectiveness.
                  </li>
                </ul>
              </div>

              {/* Project Goals */}
              <div>
                <h3 className="text-xl font-semibold mb-3">Project Goals</h3>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>
                    <strong>Performance Goals:</strong> Aim to achieve at least 85% accuracy, with precision and recall above 80%.
                  </li>
                  <li>
                    <strong>Sustainability Considerations:</strong> Optimize models to reduce computational costs and improve efficiency.
                  </li>
                  <li>
                    <strong>Ethical Considerations:</strong> Ensure data privacy, mitigate bias, and monitor fairness in model predictions.
                  </li>
                </ul>
              </div>

              {/* Expected Results */}
              <div>
                <h3 className="text-xl font-semibold mb-3">Expected Results</h3>
                <p className="text-muted-foreground">
                  We anticipate strong performance across all metrics, with supervised learning providing the best accuracy. 
                  Continuous refinement and parameter tuning will help us surpass our performance targets while ensuring an 
                  ethical and fair AI implementation.
                </p>
              </div>

            </div>
          </Card>
        </section>


        {/* References Section */}
        <section id="references" className="mb-16 scroll-mt-16">
          <Card className="p-6">
            <h2 className="text-2xl font-bold mb-4">References</h2>
            <div className="space-y-4 text-sm text-muted-foreground">
              <p>
                [1] M. Mateen, J. Wen, Nasrullah, S. Song, and Z. Huang, “Fundus image classification using VGG-19 architecture with PCA and SVD,” 
                <em>MDPI</em>, 
                <a href="https://www.mdpi.com/2073-8994/11/1/1" className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
                  https://www.mdpi.com/2073-8994/11/1/1
                </a> (accessed Feb. 20, 2025).
              </p>
              <p>
                [2] B. Mandal, A. Okeukwu, and Y. Theis, “Masked face recognition using ResNet-50,” 
                <em>arXiv.org</em>, 
                <a href="https://arxiv.org/abs/2104.08997" className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
                  https://arxiv.org/abs/2104.08997
                </a> (accessed Feb. 20, 2025).
              </p>
              <p>
                [3] B. Li, “Facial expression recognition by DenseNet-121,” 
                <em>Multi-Chaos, Fractal and Multi-Fractional Artificial Intelligence of Different Complex Systems</em>, 
                <a href="https://www.sciencedirect.com/science/article/abs/pii/B9780323900324000195" className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
                  https://www.sciencedirect.com/science/article/abs/pii/B9780323900324000195
                </a> (accessed Feb. 21, 2025).
              </p>
            </div>
          </Card>
        </section>

        {/* Additional Information Section */}
<section id="additional-requirements" className="mb-16 scroll-mt-16">
  <Card className="p-6">
    <h2 className="text-2xl font-bold mb-4">Additional Information</h2>
    <div className="space-y-6">
      
      {/* Gantt Chart */}
<div>
  <h3 className="text-xl font-semibold mb-3">Gantt Chart</h3>
  <div className="overflow-x-auto">
    <table className="w-full border border-gray-300">
      <thead>
        <tr className="bg-gray-200">
          <th className="border border-gray-300 px-4 py-2 text-left">Task</th>
          <th className="border border-gray-300 px-4 py-2 text-left">Assigned To</th>
          <th className="border border-gray-300 px-4 py-2 text-left">Start Date</th>
          <th className="border border-gray-300 px-4 py-2 text-left">End Date</th>
          <th className="border border-gray-300 px-4 py-2 text-left">Completion</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Introduction & Background</td>
          <td className="border border-gray-300 px-4 py-2">Nannan Aravazhi</td>
          <td className="border border-gray-300 px-4 py-2">2/14/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Problem Definition</td>
          <td className="border border-gray-300 px-4 py-2">Pranavkrishna Suresh</td>
          <td className="border border-gray-300 px-4 py-2">2/15/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Methods</td>
          <td className="border border-gray-300 px-4 py-2">Mihir Balsara, Gabriel Ferrer</td>
          <td className="border border-gray-300 px-4 py-2">2/16/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Potential Dataset</td>
          <td className="border border-gray-300 px-4 py-2">Henry Lin</td>
          <td className="border border-gray-300 px-4 py-2">2/16/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Potential Results & Discussion</td>
          <td className="border border-gray-300 px-4 py-2">All</td>
          <td className="border border-gray-300 px-4 py-2">2/20/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Video Creation & Recording</td>
          <td className="border border-gray-300 px-4 py-2">All</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">GitHub Page</td>
          <td className="border border-gray-300 px-4 py-2">Mihir Balsara</td>
          <td className="border border-gray-300 px-4 py-2">2/20/2024</td>
          <td className="border border-gray-300 px-4 py-2">2/21/2024</td>
          <td className="border border-gray-300 px-4 py-2">1</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>


      {/* Contribution Table */}
<div>
  <h3 className="text-xl font-semibold mb-3">Contribution Table</h3>
  <div className="overflow-x-auto">
    <table className="w-full border border-gray-300">
      <thead>
        <tr className="bg-gray-200">
          <th className="border border-gray-300 px-4 py-2 text-left">Name</th>
          <th className="border border-gray-300 px-4 py-2 text-left">Proposal Contributions</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Nannan Aravazhi</td>
          <td className="border border-gray-300 px-4 py-2">Step 1, Step 5, Gantt Chart, Contribution Table</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Pranavkrishna Suresh</td>
          <td className="border border-gray-300 px-4 py-2">Step 2, Website Development, Hosting Github Pages, Logistics</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Mihir Balsara</td>
          <td className="border border-gray-300 px-4 py-2">Step 4, Hosting Github Pages, Logistics</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Gabriel Ferrer</td>
          <td className="border border-gray-300 px-4 py-2">Step 3, Script, Creating Slides</td>
        </tr>
        <tr>
          <td className="border border-gray-300 px-4 py-2">Henry Lin</td>
          <td className="border border-gray-300 px-4 py-2">Step 3, Creating Slides, Gantt Chart</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>


      {/* Video Presentation */}
      <div>
        <h3 className="text-xl font-semibold mb-3">Video Presentation</h3>
        <p className="text-muted-foreground">
          A 3-minute recorded proposal summary using Microsoft PowerPoint, Google Slides, or equivalent. The video must be uploaded as a YouTube Unlisted video.
        </p>
        <Link href="#" className="text-primary hover:underline">
          Attach YouTube Video Link Here
        </Link>
      </div>

      {/* GitHub Repository */}
      <div>
        <h3 className="text-xl font-semibold mb-3">GitHub Repository</h3>
        <p className="text-muted-foreground">
          Here is the link to our Github repository:
        </p>
        <Link href="#" className="text-primary hover:underline">
        https://github.gatech.edu/mbalsara3/mlproposal
        </Link>
      </div>

      {/* Project Award Eligibility */}
      <div>
        <h3 className="text-xl font-semibold mb-3">Project Award Eligibility</h3>
        <p className="text-muted-foreground">
          Yes, we are interested in being considered for the Project Award.
        </p>
      </div>

    </div>
  </Card>
</section>



        {/* Scroll to top button */}
        <Button
          variant="outline"
          size="icon"
          className="fixed bottom-4 right-4"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        >
          <ArrowUpCircle className="h-4 w-4" />
        </Button>
      </main>
    </div>
  )
}

