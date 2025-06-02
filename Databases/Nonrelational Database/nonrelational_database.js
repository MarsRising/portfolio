//Create Database
use D597_Task_2
//create medical collection
const medicaldata = require("C:/Users/WGU/Data Management/medical.json")
db.medical.insertMany(medicaldata)
//See databases
show dbs
const trackerdata = require("C:/Users/WGU/Data Management/fitness_tracker.json")
db.fitness_data.insertMany(trackerdata)
//see collections
show collections
//see medical doc count
db.medical.countDocuments()
//see fitness_tracker doc count
db.fitness_tracker.countDocuments()
//check data
db.medical.find().limit(5)
//check data
db.fitness_tracker().limit(5)

//Query for the number of people who have medical conditions grouped by gender
db.medical.aggregate([
  {
    $match: {medical_conditions: {$ne:"None"}}
  },
  {
    $group: {
      _id: {medical_conditions: "$medical_conditions", gender: "$gender"},
      total: {$sum: 1}
    }
  },
  {
    $sort: {total: -1}
  }
])
//create index for optimization
db.medical.createIndex({medical_conditions: 1, gender 1})

//count of patient's who are born year 2000 and after by gender
db.medical.aggregate([
  {
    $match: {date_of_birth: { $gte: "12/31/1999" }}
  },
  {
    $group: {
      _id: "$gender",
      total: { $sum: 1 }
    }
  }
]);

//create index for optimization
db.medical.createIndex({date_of_birth: 1,gender: 1})

//total trackers by medical conditions
db.medical.aggregate([
  {
    $match: { trackers: { $ne: "None" }, medical_conditions: { $ne: "None" }}
  },
  {
    $group: {_id: "$medical_conditions", totalTrackers: { $sum: 1 }}
  },
  {
    $sort: { totalTrackers: -1 } 
  }
]);

//index for optimization
db.medical.createIndex({trackers: 1, medical_0conditions: 1});
