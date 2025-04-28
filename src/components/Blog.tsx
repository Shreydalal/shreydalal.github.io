
import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter } from '@/components/ui/card';

const Blog = () => {
  const posts = [
    {
      title: "Implementing RAG Systems: A Practical Guide for Businesses",
      date: "April 25, 2025",
      excerpt: "Learn how Retrieval Augmented Generation can transform your document processing and knowledge management, with real-world implementation strategies and best practices.",
      readTime: "8 min read",
      image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?auto=format&fit=crop&w=800&q=80"
    },
    {
      title: "The ROI of AI Automation: Real Business Impact",
      date: "April 20, 2025",
      excerpt: "Discover how modern businesses are achieving 40%+ efficiency gains through strategic AI automation. Case studies and implementation guides included.",
      readTime: "6 min read",
      image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=800&q=80"
    }
  ];

  return (
    <section id="blog" className="py-24 px-6 md:px-12 lg:px-24 bg-slate-50">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">AI Tips & Insights</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
          <p className="text-lg text-slate-600">
            Expert insights on AI automation, RAG systems, and business optimization
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {posts.map((post, index) => (
            <Card key={index} className="overflow-hidden">
              <div className="h-48 overflow-hidden">
                <img 
                  src={post.image}
                  alt={post.title}
                  className="w-full h-full object-cover transition-transform duration-500 hover:scale-105"
                />
              </div>
              <CardContent className="p-6">
                <div className="flex justify-between items-center text-sm text-slate-600 mb-3">
                  <span>{post.date}</span>
                  <span>{post.readTime}</span>
                </div>
                <h3 className="text-xl font-bold mb-3 text-slate-800">{post.title}</h3>
                <p className="text-slate-600">{post.excerpt}</p>
              </CardContent>
              <CardFooter className="p-6 pt-0">
                <Button variant="link" className="ml-auto text-blue-600 hover:text-blue-800">
                  Read More <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Blog;
