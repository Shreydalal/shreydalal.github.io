
import React from 'react';
import { MessageSquare, Code, Database, Briefcase, BarChart, ShieldCheck } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

const Services = () => {
  const services = [
    {
      icon: Briefcase,
      title: "AI Automation for Business Workflows",
      description: "Transform your manual processes into efficient, automated workflows. Reduce errors, save time, and allow your team to focus on high-value tasks that drive growth."
    },
    {
      icon: MessageSquare,
      title: "Custom AI Chatbot Development",
      description: "Intelligent conversational agents that understand your business context. From customer service to sales and internal knowledge management, deploying 24/7 AI assistance."
    },
    {
      icon: Database,
      title: "Retrieval-Augmented Generation (RAG) Solutions",
      description: "Leverage your company's documents and data to create AI systems that provide accurate, contextual responses based on your specific business knowledge."
    },
    {
      icon: Code,
      title: "CRM & Lead Management Automation",
      description: "Enhance customer relationships with AI-powered CRM solutions that intelligently qualify leads, automate follow-ups, and provide data-driven insights to boost conversion rates."
    },
    {
      icon: BarChart,
      title: "Data Analysis and Reporting using AI",
      description: "Transform raw data into actionable business intelligence. Automated reporting systems that highlight key trends, anomalies, and opportunities hidden in your business data."
    },
    {
      icon: ShieldCheck,
      title: "AI-Powered Compliance & Risk Monitoring",
      description: "Automatically monitor regulatory compliance and detect potential risks using AI. Stay ahead of audits, minimize legal exposure, and ensure your operations align with industry standards."
    }
    
  ];

  return (
    <section id="services" className="py-24 px-6 md:px-12 lg:px-24">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Services</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
          <p className="text-lg text-slate-600">
            Strategic AI solutions designed to solve specific business challenges and drive measurable results
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <Card 
              key={index} 
              className="border border-slate-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden group"
            >
              <div className="h-2 bg-blue-600 w-full group-hover:bg-blue-700 transition-colors"></div>
              <CardContent className="p-6 pt-8">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-5 text-blue-700">
                  <service.icon size={24} />
                </div>
                <h3 className="text-xl font-bold mb-3 text-slate-800 group-hover:text-blue-700 transition-colors">
                  {service.title}
                </h3>
                <p className="text-slate-600">
                  {service.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Services;
