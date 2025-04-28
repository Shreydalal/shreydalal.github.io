
import React from 'react';
import { Star } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

const Testimonials = () => {
  const testimonials = [
    {
      name: "Sarah Johnson",
      position: "CTO, LegalTech Solutions",
      company: "LegalTech Solutions",
      content: "The PDF chatbot system has completely transformed how our firm handles case research. What used to take hours now takes minutes. The AI not only finds the right information but understands the legal context behind our questions.",
      stars: 5,
      image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=200&h=200&q=80",
    },
    {
      name: "Michael Chen",
      position: "Founder, eRetail Experts",
      company: "eRetail Experts",
      content: "Working with this AI expert was game-changing for our customer service team. The workflow automation system they built has exceptional accuracy and the ability to handle complex customer interactions. Our team can now focus on the most valuable customer relationships.",
      stars: 5,
      image: "https://images.unsplash.com/photo-1560250097-0b93528c311a?auto=format&fit=crop&w=200&h=200&q=80",
    },
    {
      name: "Dr. Rebecca Torres",
      position: "Medical Director",
      company: "Horizon Healthcare",
      content: "The clinical decision graph tool has made a measurable impact on our treatment planning process. It's rare to find an AI specialist who understands both the technical requirements and the nuanced healthcare context. Highly recommended for any healthcare organization looking to leverage AI.",
      stars: 5,
      image: "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=200&h=200&q=80",
    },
  ];

  return (
    <section id="testimonials" className="py-24 px-6 md:px-12 lg:px-24">
      <div className="container mx-auto">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Client Testimonials</h2>
          <div className="w-16 h-1 bg-blue-600 mx-auto mb-6"></div>
          <p className="text-lg text-slate-600">
            Hear what clients have to say about working with me and the results we've achieved together
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card 
              key={index} 
              className="border border-slate-200 shadow-md hover:shadow-lg transition-shadow"
            >
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  {[...Array(testimonial.stars)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                
                <p className="text-slate-700 mb-6 italic">"{testimonial.content}"</p>
                
                <div className="flex items-center">
                  <div className="w-12 h-12 rounded-full overflow-hidden mr-4">
                    <img 
                      src={testimonial.image} 
                      alt={testimonial.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-900">{testimonial.name}</h4>
                    <p className="text-sm text-slate-600">{testimonial.position}, {testimonial.company}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;
