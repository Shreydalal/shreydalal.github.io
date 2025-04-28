import React from 'react';
import { FileText, Github, Linkedin } from 'lucide-react';
import { Button } from '@/components/ui/button';

const AboutMe = () => {
  const skills = [
    'LangChain', 'FastAPI', 'MongoDB', 'Pinecone', 'OpenAI API', 
    'Google Gemini', 'Python', 'TypeScript', 'React', 'Vector Databases', 
    'RAG Systems', 'LLM Fine-tuning'
  ];

  return (
    <section id="about" className="py-24 px-6 md:px-12 lg:px-24 gradient-bg">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">About Me</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <p className="text-lg text-slate-700">
              I'm an <strong className="text-blue-700">AI solutions architect</strong> with deep expertise in 
              building practical AI tools that solve real business problems. With over 5 years in the AI industry, 
              I specialize in creating systems that turn the promise of AI into tangible business outcomes.
            </p>
            
            <p className="text-lg text-slate-700">
              My approach combines technical expertise with a keen understanding of business needs.
              I focus on developing <strong className="text-blue-700">RAG models, intelligent chatbots, and automation systems</strong> that 
              reduce manual workloads, improve customer experiences, and unlock valuable insights from your data.
            </p>
            
            <p className="text-lg text-slate-700">
              Every solution I build is designed to be practical, adaptable, and deliver clear ROI. 
              I pride myself on clear communication throughout the development process and building systems 
              that grow with your business.
            </p>
            
            <div className="pt-4 flex flex-wrap gap-4">
              <Button
                className="bg-blue-600 hover:bg-blue-700 text-white"
                onClick={() => window.alert('Resume download would start here')}
              >
                <FileText className="mr-2 h-4 w-4" />
                Download Resume
              </Button>
              
              <Button variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-50">
                <Linkedin className="mr-2 h-4 w-4" />
                LinkedIn
              </Button>
              
              <Button variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-50">
                <Github className="mr-2 h-4 w-4" />
                GitHub
              </Button>
            </div>
          </div>
          
          <div>
            <div className="bg-white rounded-xl shadow-md p-8">
              <h3 className="text-xl font-bold text-blue-800 mb-6">Technical Skills</h3>
              
              <div className="flex flex-wrap gap-2">
                {skills.map((skill) => (
                  <span 
                    key={skill}
                    className="px-3 py-1 bg-blue-50 text-blue-800 rounded-full text-sm font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
              
              <hr className="my-6 border-slate-200" />
              
              <h3 className="text-xl font-bold text-blue-800 mb-4">Core Capabilities</h3>
              
              <ul className="space-y-3">
                <li className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center mt-1">
                    <div className="w-2 h-2 rounded-full bg-blue-600"></div>
                  </div>
                  <span className="ml-3 text-slate-700">Building production-ready RAG solutions</span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center mt-1">
                    <div className="w-2 h-2 rounded-full bg-blue-600"></div>
                  </div>
                  <span className="ml-3 text-slate-700">Workflow automation via AI integration</span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center mt-1">
                    <div className="w-2 h-2 rounded-full bg-blue-600"></div>
                  </div>
                  <span className="ml-3 text-slate-700">Contextual AI assistants and chatbots</span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center mt-1">
                    <div className="w-2 h-2 rounded-full bg-blue-600"></div>
                  </div>
                  <span className="ml-3 text-slate-700">Fine-tuning LLMs for specific business domains</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutMe;
