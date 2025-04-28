
import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';

const Newsletter = () => {
  const { toast } = useToast();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Subscribed!",
      description: "You'll receive your first AI tips newsletter soon.",
    });
    (e.target as HTMLFormElement).reset();
  };

  return (
    <section className="py-16 px-6 md:px-12 lg:px-24 bg-blue-50">
      <div className="container mx-auto">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-2xl md:text-3xl font-bold mb-4">Get Weekly AI Tips to Grow Your Business</h2>
          <p className="text-slate-600 mb-8">
            Join industry leaders getting actionable insights on AI automation, RAG systems, and workflow optimization.
          </p>
          
          <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-4">
            <Input
              type="email"
              placeholder="Enter your email"
              required
              className="flex-1"
            />
            <Button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white">
              Subscribe
            </Button>
          </form>
        </div>
      </div>
    </section>
  );
};

export default Newsletter;
