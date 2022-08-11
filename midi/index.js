import {GlobalKeyboardListener} from "node-global-key-listener";
const v = new GlobalKeyboardListener();
import easymidi from 'easymidi';


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