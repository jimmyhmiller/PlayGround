import React from "react";
import PropTypes from "prop-types";
import Head from 'next/head'
import tinycolor from "tinycolor2"


const GlobalStyles = () =>
  <style jsx global>{`
    body {
      margin-top: 20px;
      font-family: "Roboto", "Helvetica", "Arial", sans-serif;
      font-size: 20px;
    }
    button {
      font-family: "Roboto", "Helvetica", "Arial", sans-serif;
      font-size: 14px;
    }
    h1 {
      font-size: 64px;
      margin: 20px;
    }
    h3 {
      font-size: 24px;
      margin: 20px;
    }

    .flex {
      display: flex;
    }

    .flex.flex-row {
      flex-direction: row; 
    }

    .flex.flex-column {
      flex-direction: column; 
    }

    .next-to {
      display: flex;
    }

    @media (max-width: 1000px) {
      .flex.flex-row {
        flex-direction: column; 
      }
    }

  `}
  </style>


const Flex = ({ children, direction="default", justify, align, className="flex", style={} }) => 
  <div className={className + " flex-" + direction} style={{
    justifyContent: justify,
    alignItems: align,
    ...style,
  }}>
    {children}
  </div>

const Container = ({style={}, ...props}) => 
  <Flex 
    style={{
      "flex": 1,
      margin: "auto",
      ...style,
    }}
    {...props} />

const Heading1 = ({ text, align="left" }) => 
  <h1 style={{textAlign: align}}>
    {text}
  </h1>

const Heading3 = ({ text, align="left", style={} }) => 
  <h3 style={{
      textAlign: align,
      ...style
    }}>
    {text}
  </h3>

const Text = ({ children, align="left", secondary, style={} }) => 
  <p style={{
    textAlign: align,
    color: secondary ? "rgba(0, 0, 0, 0.54)" : "#000",
    ...style,
  }}>
    {children}
  </p>

const Card = ({ children, accentColor }) => 
  <div
    style={{
      borderTop: `5px ${accentColor} solid`,
      width: 230,
      margin: 10,
      boxShadow: "0px 1px 3px 0px rgba(0, 0, 0, 0.2), 0px 1px 1px 0px rgba(0, 0, 0, 0.14), 0px 2px 1px -1px rgba(0, 0, 0, 0.12)",
      borderRadius: 5,
      fontSize: 16,
    }}
  >
    <Flex direction="column" justify="center" style={{padding:20}}>
      {children}
    </Flex>
  </div>

const Button = ({ children, color, filled }) => {
  const backgroundColor = filled ? color : "white";
  const hoverColor = filled ? 
                    tinycolor(backgroundColor).darken().toString()
                    : tinycolor(color).setAlpha(0.2).toString()
  return (
    <div>
      <button
        style={{
          color: filled ? "white" : color,
          border: `1px ${color} solid`,
          borderRadius: 4,
          width: "100%",
          height: 30,
        }}
      >
        {children}
      </button>
      <style jsx>{`
        button {
          background-color: ${backgroundColor}
        }
        button:hover {
          background-color: ${hoverColor};
        }
      `}</style>
    </div>
  )
}

const PriceCard = ({
  title,
  subtitle,
  buttonText,
  buttonColor,
  accentColor,
  features,
  price,
  buttonFilled = false
}) => (
  <Flex>
    <Card accentColor={accentColor}>
      <Heading3 style={{ margin: 0 }} text={title} align="center" />
      {subtitle && (
        <Text style={{ margin: 0 }} secondary align="center">
          {subtitle}
        </Text>
      )}
      <div>
        <Flex direction="row" justify="center" align="center">
          <Text
            style={{ marginTop: 5, marginBottom: 0, fontSize: 32 }}
            align="center"
          >
            ${price}<span style={{fontSize:20}}>/mo</span>
          </Text>
        </Flex>
        <div style={{margin: 10}}>
          {features.map(feature => 
            <Text style={{marginBottom: 5, marginTop: 0}} variant="subtitle1" align="center" key={feature}>
              {feature}
            </Text>
          )}
        </div>
      </div>
      <div>
        <Button color={buttonColor || accentColor} filled={buttonFilled}>
          {buttonText}
        </Button>
      </div>
    </Card>
  </Flex>
);
const Pricing = () =>
  <Flex direction="row" justify="center" align="flex-end">
    <PriceCard
      price={0}
      features={["Non-Commercial Use", "Up to 10 users"]}
      accentColor="rgb(57 104 178)"
      buttonText="Add To Slack"
      title="Personal" />
    <PriceCard 
      price={15}
      features={["50 polls a month", "Unlimited Users", "30 day free trial"]}
      buttonFilled={true}
      accentColor="#fb9353"
      subtitle="Most Popular"
      buttonText="Try Now - No sign up"
      title="Basic" />
    <PriceCard
      price={25}
      features={["100 polls a month", "Unlimited Users"]}
      accentColor="rgb(83 166 251)"
      buttonVariant="contained"
      buttonText="Sign Up Now"
      title="Premium" />
    <PriceCard
      price={50}
      features={["Unlimited polls a month", "Unlimited Users"]}
      accentColor="rgb(63, 140, 251)"
      buttonText="Sign Up Now"
      title="Enterprise" />
  </Flex>

export default (props) =>
  <>
    <Head>
      <title>Poll App - Slack polls made easy</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    </Head>
    <GlobalStyles />
    <Container direction="column" justify="center">
      <Flex direction="row" justify="center">
        <Flex direction="column" justify="center">
          <Flex className="next-to" direction="row" align="baseline" justify="center">
            <img style={{width: 75, height: 75 }} src="/static/logo.png" />
            <Heading1
              align="center"
              text="Poll App" />
          </Flex>
          <Flex justify="center">
            <Text secondary align="center" style={{maxWidth: 600}}>
              Make and take polls right in Slack. Gather feedback or make decisions without needing to schedule a meeting.
            </Text>
          </Flex>
          <Flex style={{paddingTop: 0, paddingBottom: 20}} justify="center">
            <img 
              style={{cursor: "pointer"}}
              alt="Add to Slack" 
              src="https://platform.slack-edge.com/img/add_to_slack.png" 
              srcSet="https://platform.slack-edge.com/img/add_to_slack.png 1x, https://platform.slack-edge.com/img/add_to_slack@2x.png 2x" 
              width="139" 
              height="40" />
          </Flex>
        </Flex>

        <Flex direction="column" justify="center" align="center">
          <img style={{width: 253, height: 500}} src="/static/iphone-white.png" />
        </Flex>
      </Flex>
    </Container>
    <Container justify="center" style={{paddingTop: 30}}>
      <Pricing />
    </Container>
  </>
