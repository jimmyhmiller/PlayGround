import {GlobalKeyboardListener} from "node-global-key-listener";
const v = new GlobalKeyboardListener();
import easymidi from 'easymidi';



const oldFunction = () => {
    const virtualOutput = new easymidi.Output('Virtual output name', true);





    let number = 0;
    let step = 1;
    let defaultStep = 1;

    let lastAction = performance.now()




    //Log every key that's pressed.
    v.addListener(function (e, down) {


        if (e.state === "DOWN" || 
                (e.name !== "F17" && e.name !== "F18" && e.name !== "F19" && e.name !== "F20")) {
            return false;
        }



        let nextAction = performance.now();
        let diff = nextAction - lastAction
        lastAction = nextAction


        if (diff < 80) {
            step = defaultStep * 5;
        } else {
            step = defaultStep
        }

        console.log(diff, step, number, e.name)


        if (e.name === "F17") {
            number = 0
            defaultStep -= 1;
            virtualOutput.send('cc', {
              controller: 0,
              value: number,
              channel: 0
            })
        }

        if (e.name === "F18") {
            number = 0
            defaultStep += 1;
            virtualOutput.send('cc', {
              controller: 0,
              value: step,
              channel: 0
            })
        }

        if (e.name === "F19") {
            number -= step
            virtualOutput.send('cc', {
              controller: 0,
              value: step * -1,
              channel: 0
            })
        }




        if (e.name === "F20") {
            number += step
            // number = number % 316;
            virtualOutput.send('cc', {
              controller: 0,
              value: step,
              channel: 0
            })
        }
    });
}

import robot from 'robotjs'



const midiTwo = () => {

    const input = new easymidi.Input('ortho remote Bluetooth');
    const output = new easymidi.Output('ortho remote Bluetooth');

    const myGenerator = function*() {
        while (true) {
            yield
            let direction = yield;
            if (direction.value === "increasing") {
                robot.keyTap("f10")
            }
        }
    }
    const increasing = function*() {
        let status = "increasing";
        let lastValue = Number.MIN_SAFE_INTEGER;
        while (true) {
            const value = yield status;
            if (value > lastValue || value === 0 && lastValue > 120) {
                status = "increasing"
            } else {
                status = "decreasing"
            }
            lastValue = value;
        }
    }


    let generator = myGenerator();
    let increaser = increasing();
    input.on("cc", e => {
        console.log(e)
        if (e.value === 0) {
            output.send('sysex', [0xF0, 0x00, 0x20, 0x76, 0x02, 0x00, 0x06, 0x7F, 0xF7]);
        } else if (e.value === 127) {
             output.send('sysex', [0xF0, 0x00, 0x20, 0x76, 0x02, 0x00, 0x06, 0x00, 0xF7]);
        }
        let direction = increaser.next(e.value)
        generator.next(direction)
    })

    input.on("noteon", e => {
        robot.keyTap("f11")
    })






    console.log(easymidi.getInputs())

}


midiTwo();