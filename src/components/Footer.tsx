import React from 'react';
import { Linkedin, Github, Mail } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-slate-900 text-slate-200 py-12 px-6">
      <div className="container mx-auto">
        <div className="grid md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <h3 className="text-xl font-bold mb-4 text-white">AI Expert</h3>
            <p className="text-slate-400 max-w-md">
              Building intelligent AI solutions for forward-thinking businesses.
              Automation, chatbots, RAG systems, and custom AI applications that 
              deliver measurable business value.
            </p>
            
            <div className="flex gap-4 mt-6">
              <a 
                href="#" 
                className="w-8 h-8 rounded-full bg-slate-800 hover:bg-blue-600 flex items-center justify-center text-white transition-colors"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-4 h-4" />
              </a>
              <a 
                href="#" 
                className="w-8 h-8 rounded-full bg-slate-800 hover:bg-blue-600 flex items-center justify-center text-white transition-colors"
                aria-label="GitHub"
              >
                <Github className="w-4 h-4" />
              </a>
              <a 
                href="mailto:contact@aiexpert.com" 
                className="w-8 h-8 rounded-full bg-slate-800 hover:bg-blue-600 flex items-center justify-center text-white transition-colors"
                aria-label="Email"
              >
                <Mail className="w-4 h-4" />
              </a>
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4 text-white">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <a href="#about" className="text-slate-400 hover:text-white transition-colors">About</a>
              </li>
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">Services</a>
              </li>
              <li>
                <a href="#portfolio" className="text-slate-400 hover:text-white transition-colors">Portfolio</a>
              </li>
              <li>
                <a href="#testimonials" className="text-slate-400 hover:text-white transition-colors">Testimonials</a>
              </li>
              <li>
                <a href="#contact" className="text-slate-400 hover:text-white transition-colors">Contact</a>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4 text-white">Services</h4>
            <ul className="space-y-2">
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">AI Automation</a>
              </li>
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">Custom Chatbots</a>
              </li>
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">RAG Solutions</a>
              </li>
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">CRM Automation</a>
              </li>
              <li>
                <a href="#services" className="text-slate-400 hover:text-white transition-colors">AI Data Analysis</a>
              </li>
            </ul>
          </div>
        </div>
        
        <div className="mt-12 pt-8 border-t border-slate-800 text-center text-slate-500">
          <p>© {new Date().getFullYear()} AI Expert. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
