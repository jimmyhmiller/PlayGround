// Based off https://skorokithakis.github.io/expounder/
// I'd love to move this into an actual library. I really like the idea.

import React, { Component } from 'react';

class Link extends Component {
    show = () => {
        console.log('show');
        this.context.show(this.props.expound);
    }

    render() {
        return (
            <span>
                {' '}
                <span style={{borderBottom: "1px dashed"}} onClick={this.show}>{this.props.children}</span>
                {' '}
            </span>
        );
    }
}

Link.contextTypes = {
  show: React.PropTypes.func
};


class ExpoundExtern extends Component {
    render() {
        let display = this.context.opened.includes(this.props.expound) ? 'inline' : 'none'
        return (
            <span style={{display}}>{this.props.children}</span>
        );
    }
}

ExpoundExtern.contextTypes = {
  opened: React.PropTypes.array
};


class Expoundable extends Component {
    state = {
        opened: []
    }

    getChildContext() {
        return {
            opened: this.state.opened,
            show: this.addOpen,
        }
    }

    addOpen = (link) => {
        this.setState({
            opened: this.state.opened.concat(link)
        })
    }

    render() {
        return (
            <span>{this.props.children}</span>
        );
    }
}

Expoundable.childContextTypes = {
  show: React.PropTypes.func,
  opened: React.PropTypes.array,
};


class ExpoundSelf extends Component {

    state = {
        hidden: true
    }

    show = () => {
        this.setState({
            hidden: false
        })
    }

    render() {
        let display = this.state.hidden ? 'none' : 'inline' 
        return (
            <span>
                <span style={{borderBottom: "1px dashed"}} onClick={this.show}>{this.props.title}</span>
                <span style={{display}}>{this.props.children}</span>
            </span>
        );
    }
}



class Expound extends Component {
    render() {
        if (this.props.title) {
            return <ExpoundSelf {...this.props} />
        } else {
            return <ExpoundExtern {...this.props} />
        }
    }
}

class ExpoundApp extends Component {
    render() {
        return (
            <span>
                <Expoundable>
                    As was said by
                    <Link expound="plantinga">Alvin Plantinga,</Link>
                    "I like Pizza<Expound expound="plantinga"> a whole lot</Expound>"
                    
                    <Link expound="test"> Test</Link>
                    <Expound expound="test"> EXPOUNDED!</Expound>
                    <Expound expound="test"> EXPOUNDED Twice!</Expound>
                </Expoundable>

            </span>
        );
    }
}


export default ExpoundApp;