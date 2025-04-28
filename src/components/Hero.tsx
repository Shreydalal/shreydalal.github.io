
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Hero = () => {
  return (
    <section className="min-h-[90vh] flex items-center justify-center pt-20 pb-16 px-6 md:px-12 lg:px-24 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-20 left-10 w-64 h-64 bg-blue-200/20 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-72 h-72 bg-blue-300/20 rounded-full filter blur-3xl"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMzYjgyZjYxMCIgZmlsbC1vcGFjaXR5PSIwLjA1Ij48cGF0aCBkPSJNMzYgMzRoLTJWMTZoLTJ2MmgtMnYyaC0ydjJoLTJ2MmgtMnYyaC0ydjJIMjB2MmgtMnYyaC0ydjJIMHYyaDJ2MmgydjJoMnYyaDJ2MmgydjJoMnYtMkgydjJoMnYyaDJ2MmgydjJoMnYySDB2Mmg0djJIMHYyaDR2LTJoMnYtMmgydi0yaDJ2MmgydjJIMjAiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-80"></div>
      </div>

      <div className="container mx-auto flex flex-col lg:flex-row items-center justify-between gap-16 z-10">
        <div className="flex-1 text-center lg:text-left">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-6">
            I Build <span className="text-gradient">AI Solutions</span> That Automate, Accelerate, and Empower Businesses
          </h1>
          
          <p className="text-slate-600 text-lg md:text-xl mb-10 max-w-2xl mx-auto lg:mx-0">
            Helping companies save time and increase efficiency through custom chatbots, 
            RAG implementations, and intelligent workflow automation.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
            <Button 
              size="lg" 
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 text-lg"
              onClick={() => window.location.href = '#contact'}
            >
              Work With Me <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            
            <Button 
              variant="outline" 
              size="lg" 
              className="border-blue-600 text-blue-600 hover:bg-blue-50 px-8 py-6 text-lg"
              onClick={() => window.location.href = '#portfolio'}
            >
              See My Work
            </Button>
          </div>
        </div>
        
        <div className="flex-1 w-full max-w-md">
          <div className="relative">
            {/* Simulated 3D Floating AI Graphic */}
            <div className="relative w-full aspect-square max-w-md mx-auto">
              <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-blue-600/20 to-blue-400/20 animate-pulse"></div>
              <div className="absolute inset-4 rounded-full bg-gradient-to-br from-blue-500/30 to-blue-300/30"></div>
              <div className="absolute inset-8 rounded-full bg-white shadow-lg flex items-center justify-center">
                <svg className="w-2/3 h-2/3 text-blue-500" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 16V7.2C21 6.0799 21 5.51984 20.782 5.09202C20.5903 4.71569 20.2843 4.40973 19.908 4.21799C19.4802 4 18.9201 4 17.8 4H6.2C5.07989 4 4.51984 4 4.09202 4.21799C3.71569 4.40973 3.40973 4.71569 3.21799 5.09202C3 5.51984 3 6.0799 3 7.2V16M3 16L10 12L14 14.5L21 10.5M3 16C3 17.1201 3 17.6802 3.21799 18.108C3.40973 18.4843 3.71569 18.7903 4.09202 18.982C4.51984 19.2 5.07989 19.2 6.2 19.2H17.8C18.9201 19.2 19.4802 19.2 19.908 18.982C20.2843 18.7903 20.5903 18.4843 20.782 18.108C21 17.6802 21 17.1201 21 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M8 8H10M8 11H12M10.5 15C10.5 15 9 15 9 13.5C9 12 10.5 12 10.5 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
            </div>
            
            {/* Network Connection Lines */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M150 50 L70 120" stroke="rgba(59, 130, 246, 0.3)" strokeWidth="1" strokeDasharray="4 4"/>
              <path d="M150 50 L230 120" stroke="rgba(59, 130, 246, 0.3)" strokeWidth="1" strokeDasharray="4 4"/>
              <path d="M70 120 L150 200" stroke="rgba(59, 130, 246, 0.3)" strokeWidth="1" strokeDasharray="4 4"/>
              <path d="M230 120 L150 200" stroke="rgba(59, 130, 246, 0.3)" strokeWidth="1" strokeDasharray="4 4"/>
              <circle cx="150" cy="50" r="8" fill="rgba(59, 130, 246, 0.6)"/>
              <circle cx="70" cy="120" r="8" fill="rgba(59, 130, 246, 0.6)"/>
              <circle cx="230" cy="120" r="8" fill="rgba(59, 130, 246, 0.6)"/>
              <circle cx="150" cy="200" r="8" fill="rgba(59, 130, 246, 0.6)"/>
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
