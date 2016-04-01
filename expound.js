// Based off https://skorokithakis.github.io/expounder/
// I'd love to move this into an actual library. I really like the idea.


class Link extends Component {
    show = () => {
        console.log('show');
        this.props.show(this.props.expound);
    }

    render() {
        return (
            <span>
                <span style={{borderBottom: "1px dashed"}} onClick={this.show}>{this.props.children}</span>
            </span>
        );
    }
}

class ExpoundExtern extends Component {
    render() {
        let display = this.props.opened.includes(this.props.expound) ? 'inline' : 'none'
        return (
            <span style={{display}}>{this.props.children}</span>
        );
    }
}

class Expoundable extends Component {
    state = {
        opened: []
    }

    addOpen = (link) => {
        this.setState({
            opened: this.state.opened.concat(link)
        })
    }

    transferProps = () => {
        return React.Children.map(this.props.children, (child) => {
            if (child.type === Link) {
                return React.cloneElement(child, {show: this.addOpen})
            }
            if (child.type === Expound) {
                return React.cloneElement(child, {opened: this.state.opened})
            }
            return child;
        });
    }

    render() {
        return (
            <span>{this.transferProps()}</span>
        );
    }
}


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


