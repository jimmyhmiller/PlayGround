//
//  ContentView.swift
//  Meld
//
//  Created by Jimmy Miller on 9/24/23.
//

import SwiftUI

func isEventInBounds(eventX: Int, eventY: Int, x: Int, y: Int, width: Int, height: Int) -> Bool {
    if eventX >= x && eventX <= x + width && eventY >= y && eventY <= y + height {
        return true
    } else {
        return false
    }
}

func initRects(screenSize: CGRect, num: Int) -> [(Int, Int, Int, Int)] {
    var tempRects : [(Int, Int, Int, Int)] = []
    for _ in 0...num {
        let width = Int.random(in: 50...300)
        let height = Int.random(in: 50...300)
        let x = Int.random(in: 0...Int(screenSize.width) - width)
        let y = Int.random(in: 0...Int(screenSize.height) - height)
        tempRects.append((x, y, width, height))
    }
    return tempRects
}

struct ContentView: View {
    

    
    @State var rects : [(Int, Int, Int, Int)]
    @State var touching = false
    @State var indexMoving = -1;
    @State var offset : (CGFloat, CGFloat) = (0.0, 0.0)
    @State var startOffset : (CGFloat, CGFloat) = (0.0, 0.0)
    @State var numberOfRects = 10.0
    
    init() {
        let screenSize: CGRect = UIScreen.main.bounds
        self.rects = initRects(screenSize: screenSize, num: Int(10.0))
        
    }
    var body: some View {
        Slider(value: $numberOfRects, in:0...3000, step: 5)
            .onChange(of: numberOfRects) {
                self.rects = initRects(screenSize: UIScreen.main.bounds, num: Int(numberOfRects))
            }
        ZStack {
            Color(red: 0.87, green: 0.88, blue: 0.89).ignoresSafeArea(.all)
            Canvas(
                opaque: false,
                colorMode: .linear,
                rendersAsynchronously: false
            ) { context, size in
                let shadowFilter = GraphicsContext.Filter.shadow(color: .black.opacity(0.08), radius: 5, x: 0, y: 2)
                context.addFilter(shadowFilter)
                for (i, (x, y, width, height)) in self.rects.enumerated() {

                    let rect = CGRect(origin: CGPoint(x: CGFloat(x), y: CGFloat(y)), size: CGSize(width: CGFloat(width), height: CGFloat(height)))

                    let rectangle = RoundedRectangle(cornerSize: CGSize(width: 20, height: 20))
                        .path(in: rect)

                    let color : GraphicsContext.Shading
                    if !(self.indexMoving == i) {
                        color = GraphicsContext.Shading.color(red: 0.945, green: 0.941, blue: 0.93)
                    } else {
                        color = GraphicsContext.Shading.color(.red)
                    }
                   
                    
                    context.fill(
                        rectangle,
                        with: color
                    )
                }
            }
            .gesture(DragGesture(minimumDistance: 0).onChanged { drag in
                if !self.touching {
                    for (i, (x, y, width, height)) in self.rects.enumerated().reversed() {
                        if isEventInBounds(eventX: Int(drag.startLocation.x),
                                           eventY: Int(drag.startLocation.y),
                                           x: x,
                                           y: y,
                                           width: width,
                                           height: height) {
                            
                                self.touching = true
                                self.indexMoving = i
                                
                                self.startOffset = (drag.location.x - CGFloat(self.rects[self.indexMoving].0), drag.location.y - CGFloat(self.rects[self.indexMoving].1))
                            
                               
                                break;
                            }
                        }
                } else {
                    self.rects[indexMoving] = (Int(drag.location.x - self.startOffset.0), Int(drag.location.y - self.startOffset.1), rects[indexMoving].2, rects[indexMoving].3)
                }
            }.onEnded { _ in
                self.touching = false
                self.indexMoving = -1
            })
        }
    }
}

#Preview {
    ContentView()
}
