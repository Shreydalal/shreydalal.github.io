
import React from 'react';
import Navbar from '../components/Navbar';
import Hero from '../components/Hero';
import AboutMe from '../components/AboutMe';
import Services from '../components/Services';
import Portfolio from '../components/Portfolio';
import Testimonials from '../components/Testimonials';
import Newsletter from '../components/Newsletter';
import Blog from '../components/Blog';
import Contact from '../components/Contact';
import Footer from '../components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <AboutMe />
      <Services />
      <Portfolio />
      <Testimonials />
      <Newsletter />
      <Blog />
      <Contact />
      <Footer />
    </div>
  );
};

export default Index;
