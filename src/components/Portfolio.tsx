
import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

const Portfolio = () => {
  const projects = [
    {
      title: "AI PDF Document Chatbot",
      description: "Built an intelligent document processing system that allows users to chat with their PDFs. The solution extracts, indexes, and retrieves information from complex documents.",
      problem: "A legal firm struggled with quickly accessing information from thousands of case documents.",
      solution: "Created a specialized RAG system with document chunking, vector embeddings, and context-aware responses.",
      result: "Reduced research time by 70% and improved information accuracy in client consultations.",
      technologies: ["FastAPI", "Pinecone", "Google Gemini", "LangChain", "React"],
      image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?auto=format&fit=crop&w=800&q=80"
    },
    {
      title: "Business Workflow AutoGPT",
      description: "Developed an autonomous AI agent that handles routine business tasks like email classification, data entry, and preliminary customer inquiries.",
      problem: "A growing e-commerce company was overwhelmed by repetitive customer service requests.",
      solution: "Implemented a multi-step autonomous agent that could understand, categorize, and resolve common customer issues.",
      result: "Automated response to 65% of customer inquiries with 92% satisfaction rating.",
      technologies: ["LangChain", "FastAPI", "MongoDB", "React", "AutoGPT"],
      image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=800&q=80"
    },
    {
      title: "CRM Automation for Real Estate",
      description: "Created an AI-enhanced CRM system that qualifies leads, recommends properties, and schedules viewings automatically based on customer preferences.",
      problem: "Real estate agents spent too much time on administrative tasks rather than closing deals.",
      solution: "Designed a custom CRM with AI integration for lead scoring, property matching, and automated follow-ups.",
      result: "Increased agent productivity by 40% and improved lead conversion rates by 28%.",
      technologies: ["Python", "OpenAI API", "PostgreSQL", "TypeScript", "Next.js"],
      image: "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=800&q=80"
    },
    {
      title: "Clinical Decision Graph Generator",
      description: "Built a specialized healthcare AI tool that analyzes clinical notes and generates visual decision trees to support treatment planning.",
      problem: "Physicians needed to quickly identify treatment options across complex patient cases with multiple conditions.",
      solution: "Developed an NLP system that extracts key medical factors from clinical notes and generates treatment decision graphs.",
      result: "Reduced treatment planning time by 45% and improved consistency across the medical team.",
      technologies: ["Python", "Google Gemini", "D3.js", "MongoDB", "FastAPI"],
      image: "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=800&q=80"
    }
  ];

  return (
    <section id="portfolio" className="py-24 px-6 md:px-12 lg:px-24 bg-slate-50">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Featured Projects</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
          <p className="text-lg text-slate-600">
            A selection of recent AI solutions I've developed for clients across industries
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <Card key={index} className="overflow-hidden border-none shadow-lg">
              <div className="h-48 overflow-hidden">
                <img 
                  src={project.image} 
                  alt={project.title}
                  className="w-full h-full object-cover transition-transform duration-500 hover:scale-105"
                />
              </div>
              <CardContent className="p-6">
                <h3 className="text-xl font-bold mb-3 text-blue-800">{project.title}</h3>
                <p className="text-slate-600 mb-4">{project.description}</p>
                
                <div className="mb-4 space-y-2">
                  <div>
                    <span className="font-semibold text-slate-800">The Challenge: </span>
                    <span className="text-slate-600">{project.problem}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-800">The Solution: </span>
                    <span className="text-slate-600">{project.solution}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-800">The Result: </span>
                    <span className="text-slate-600">{project.result}</span>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-2 mt-4">
                  {project.technologies.map((tech, techIndex) => (
                    <Badge key={techIndex} variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                      {tech}
                    </Badge>
                  ))}
                </div>
              </CardContent>
              <CardFooter className="bg-slate-50 p-4">
                <Button variant="ghost" className="text-blue-600 hover:text-blue-800 hover:bg-blue-50 ml-auto">
                  View Details <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Portfolio;
