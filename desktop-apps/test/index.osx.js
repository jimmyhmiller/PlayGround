/**
 * Sample React Native Desktop App
 * https://github.com/ptmt/react-native-desktop
 */
 import React from 'react';
 import ReactNative from 'react-native-desktop';
 const {
   AppRegistry,
   StyleSheet,
   Text,
   View,
   Dimensions,
 } = ReactNative;

const test = React.createClass({
  render() {
    const columnSize = {
      width: Dimensions.get('window').width / 4,
      height: Dimensions.get('window').height
    }
    return (
      <View style={styles.container}>
        <View style={[styles.firstColumn, columnSize]} />
        <View style={styles.secondColumn} />
      </View>
    );
  }
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5FCFF',
  },
  firstColumn: {
    backgroundColor: 'rgb(36, 47, 67)'
  }
});

AppRegistry.registerComponent('test', () => test);
