
import React from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

const PredefinedPackages = () => {
  const { toast } = useToast();
  
  const handleContactClick = (packageName: string) => {
    toast({
      title: `${packageName} Package Selected`,
      description: "We'll contact you shortly to discuss this package!",
    });
    
    // Scroll to contact section
    const contactSection = document.getElementById('contact');
    if (contactSection) {
      contactSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const packages = [
    {
      name: "Starter",
      price: "$2,500",
      description: "Perfect for small businesses just beginning their AI journey",
      features: [
        "Custom AI chatbot setup & integration",
        "Single workflow automation",
        "Basic document processing with RAG",
        "1-month support & maintenance",
        "Implementation report with ROI metrics"
      ],
      highlighted: false
    },
    {
      name: "Business",
      price: "$5,000",
      description: "Comprehensive solution for growing businesses with multiple AI needs",
      features: [
        "Advanced chatbot with custom knowledge base",
        "3 workflow automations",
        "Full RAG implementation with document management",
        "AI-powered analytics dashboard",
        "3-month support & maintenance",
        "Staff training sessions"
      ],
      highlighted: true
    },
    {
      name: "Enterprise",
      price: "$12,000+",
      description: "Full-scale AI transformation for larger organizations",
      features: [
        "End-to-end AI ecosystem design & implementation",
        "Unlimited workflow automations",
        "Advanced RAG with multi-source integration",
        "Custom LLM fine-tuning",
        "AI strategy consultation",
        "6-month support & dedicated consultant",
        "Quarterly optimization reviews"
      ],
      highlighted: false
    }
  ];

  return (
    <section id="packages" className="py-24 px-6 md:px-12 lg:px-24 bg-slate-50">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Packages</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
          <p className="text-lg text-slate-600">
            Choose the perfect AI solution package for your business needs
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {packages.map((pkg, index) => (
            <Card 
              key={index}
              className={`border ${pkg.highlighted ? 'border-blue-600 shadow-lg relative' : 'border-slate-200'} transition-all h-full flex flex-col`}
            >
              {pkg.highlighted && (
                <div className="absolute -top-4 left-0 right-0 mx-auto w-fit bg-blue-600 text-white px-4 py-1 rounded-full text-sm font-medium">
                  Most Popular
                </div>
              )}
              <CardHeader className={`pb-2 ${pkg.highlighted ? 'pt-8' : 'pt-6'}`}>
                <CardTitle className="text-2xl text-center mb-2">{pkg.name}</CardTitle>
                <div className="text-center">
                  <span className="text-3xl font-bold">{pkg.price}</span>
                  {pkg.name !== "Enterprise" && <span className="text-slate-500"> / project</span>}
                </div>
              </CardHeader>
              <CardContent className="flex-grow">
                <p className="text-center text-slate-600 mb-6 min-h-[48px]">
                  {pkg.description}
                </p>
                <ul className="space-y-3 mb-6">
                  {pkg.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="mr-2 mt-1 text-blue-600">
                        <Check size={18} />
                      </span>
                      <span className="text-slate-700">{feature}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter className="pt-2 pb-6">
                <Button 
                  className={`w-full ${pkg.highlighted ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'}`}
                  onClick={() => handleContactClick(pkg.name)}
                >
                  Get Started
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PredefinedPackages;
